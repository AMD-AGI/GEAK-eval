import torch
import math
from functools import partial
from typing import Tuple

try:
    # fa3
    from flash_attn_interface import flash_attn_func 
except:
    # fa2
    from flash_attn import flash_attn_func

import torch
import triton
import triton.language as tl
import math

# Fused attention kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'CHUNK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 32, 'CHUNK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'CHUNK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 16, 'CHUNK_N': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 32, 'CHUNK_N': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'CHUNK_N': 2048}, num_stages=2, num_warps=8),
    ],
    key=['N', 'D']
)
@triton.jit
def _fused_attn_fwd_kernel(P,
                           W,
                           O,
                           N:tl.constexpr,
                           D:tl.constexpr,
                           BLOCK_N:tl.constexpr,
                           CHUNK_N:tl.constexpr
                           ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1) * BLOCK_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask = off_n < N

    acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    weight_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(3):
        p = tl.load(P+i).to(tl.pointer_type(O.dtype.element_ty))
        o = tl.load(p + offset, mask=mask[:, None], other=0.).to(tl.float32)
        w = tl.load(W + off_n * 3 + i, mask=mask, other=0.).to(tl.float32)
        w_sigmoid = tl.sigmoid(w)
        weight_sum += w_sigmoid
        acc += o * w_sigmoid[:, None]
    # Normalize by weight sum to prevent accumulation errors
    acc = acc / tl.maximum(weight_sum[:, None], 1e-6)
    tl.store(O + offset, acc, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 4, 'CHUNK_N': 1024}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_N': 8, 'CHUNK_N': 1024}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_N': 16, 'CHUNK_N': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'CHUNK_N': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 4, 'CHUNK_N': 2048}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_N': 8, 'CHUNK_N': 2048}, num_stages=4, num_warps=2),
    ],
    key=['N', 'D']
)
@triton.jit
def _fused_attn_bwd_kernel(DP,
                           DW,
                           DO,
                           P,
                           W,
                           N:tl.constexpr,
                           D:tl.constexpr,
                           BLOCK_N:tl.constexpr,
                           CHUNK_N:tl.constexpr
                           ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1) * BLOCK_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask = off_n < N

    offset = off_n[:, None] * D + tl.arange(0, D)[None, :]
    dcombine_o = tl.load(DO + offset, mask=mask[:, None], other=0.).to(tl.float32)

    i = tl.program_id(2)
    p = tl.load(P+i).to(tl.pointer_type(DO.dtype.element_ty))
    dp = tl.load(DP+i).to(tl.pointer_type(DO.dtype.element_ty))
    o = tl.load(p + offset, mask=mask[:, None], other=0.).to(tl.float32)
    w = tl.load(W + off_n * 3 + i, mask=mask, other=0.).to(tl.float32)
    sigmoid_w = tl.sigmoid(w)
    do = dcombine_o * sigmoid_w[:, None]
    dsigmoid_w = tl.sum(dcombine_o * o, -1)
    dw = sigmoid_w * (1 - sigmoid_w) * dsigmoid_w
    tl.store(dp + offset, do, mask=mask[:, None])
    tl.store(DW + off_n * 3 + i, dw, mask=mask)

    

class _FusedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, w):
        assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous() and w.is_contiguous()
        B, S, H, D = a.shape
        assert w.size(-1) == 3
        assert math.log2(D).is_integer()
        o = torch.empty_like(a)
        N = B * S * H
        p = torch.tensor([a.data_ptr(), b.data_ptr(), c.data_ptr()], dtype=torch.int64, device=a.device)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_N']))
        _fused_attn_fwd_kernel[grid](p, 
                                     w, 
                                     o, 
                                     N, 
                                     D
                                     )
        ctx.save_for_backward(a, b, c, w)
        ctx.p = p
        ctx.N = N
        ctx.D = D

        return o 
    
    @staticmethod
    def backward(ctx, do):
        assert do.is_contiguous()
        a, b, c, w = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dc = torch.empty_like(c)
        dw = torch.empty_like(w)
        dp = torch.tensor([da.data_ptr(), db.data_ptr(), dc.data_ptr()], dtype=torch.int64, device=a.device)
        grid = lambda meta: (triton.cdiv(ctx.N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_N']), 3)
        _fused_attn_bwd_kernel[grid](dp, 
                                     dw,
                                     do,
                                     ctx.p,
                                     w, 
                                     ctx.N, 
                                     ctx.D
                                     )
        return da, db, dc, dw


def fused_attention(a, b, c, w):
    return _FusedAttention.apply(a, b, c, w)

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
    ],
    key=['D1','D2','BLOCK_SIZE_N']
)
@triton.jit
def _block_compress_fwd(X, W, PE, Y, 
                        x_stride_b, x_stride_n, x_stride_h, x_stride_d,
                        y_stride_b, y_stride_m, y_stride_h, y_stride_d,
                        stride, kernel_size, 
                        D,
                        D1:tl.constexpr, D2:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_m = tl.cast(tl.program_id(2), tl.int64)
    
    X += off_b * x_stride_b + off_h * x_stride_h + stride * off_m * x_stride_n
    Y += off_b * y_stride_b + off_h * y_stride_h + off_m * y_stride_m

    rows = tl.arange(0, BLOCK_SIZE_N)
    mask = rows < kernel_size

    w = tl.load(W + rows, mask=mask, other=0.).to(tl.float32)

    x_ptrs = X + rows[:, None] * x_stride_n + tl.arange(0, D1)[None, :]
    x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
    pe_ptrs = PE + rows[:, None] * D + tl.arange(0, D1)[None, :]
    pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
    # y = tl.sum((x + pe) * w[:, None], axis=0) / kernel_size
    w_sum = tl.sum(tl.where(mask, w, 0.0))
    y = tl.sum((x + pe) * w[:, None], axis=0) / tl.maximum(w_sum, 1e-6)
    y_ptrs = Y + tl.arange(0, D1)
    tl.store(y_ptrs, y)
    
    if D2 > 0:
        x_ptrs = X + rows[:, None] * x_stride_n + tl.arange(0, D2)[None, :] + D1
        x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        pe_ptrs = PE + rows[:, None] * D + tl.arange(0, D2)[None, :] + D1
        pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        # y = tl.sum((x + pe) * w[:, None], axis=0) / kernel_size
        y = tl.sum((x + pe) * w[:, None], axis=0) / tl.maximum(w_sum, 1e-6)
        y_ptrs = Y + tl.arange(0, D2) + D1
        tl.store(y_ptrs, y)

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=16),
    ],
    key=['D1','D2', 'BLOCK_SIZE_N']
)
@triton.jit
def _block_compress_dwdpe(DY, DW, DPE,
                          X, W, PE,
                          dy_stride_b, dy_stride_m, dy_stride_h, dy_stride_d,
                          x_stride_b, x_stride_n, x_stride_h, x_stride_d,
                          stride, kernel_size, num_blocks, NUM_SMS, 
                          B, H, D,
                          D1:tl.constexpr, D2:tl.constexpr, BLOCK_SIZE_N:tl.constexpr
                          ):
    pid = tl.cast(tl.program_id(0), tl.int64)
    current_id = pid
    total = B * H * num_blocks

    rows = tl.arange(0, BLOCK_SIZE_N)
    mask = rows < kernel_size
    cols = tl.arange(0, D1)

    w = tl.load(W+rows, mask=mask, other=0.).to(tl.float32)
    pe_ptrs = PE + rows[:, None] * D + cols[None, :]
    pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)

    dpe = tl.zeros((BLOCK_SIZE_N, D1), dtype=tl.float32)
    dw = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    if D2 > 0:
        cols2 = tl.arange(0, D2) + D1
        pe_ptrs2 = PE + rows[:, None] * D + cols2[None, :]
        pe2 = tl.load(pe_ptrs2, mask=mask[:, None], other=0.).to(tl.float32)
        dpe2 = tl.zeros((BLOCK_SIZE_N, D2), dtype=tl.float32)

    while current_id < total:
        off_m = current_id % num_blocks
        off_bh = current_id // num_blocks
        off_b = off_bh // H
        off_h = off_bh % H

        dy_ptrs = DY + off_b * dy_stride_b + off_h * dy_stride_h + off_m * dy_stride_m + cols
        x_ptrs = X + off_b * x_stride_b + off_h * x_stride_h \
                + (stride * off_m + rows[:, None]) * x_stride_n \
                + cols[None, :]
        dy = tl.load(dy_ptrs).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        x_pe = x + pe
        dw += tl.sum(x_pe * dy[None, :], axis=1)
        dpe += w[:, None] * dy[None, :]

        if D2 > 0:
            dy_ptrs2 = DY + off_b * dy_stride_b + off_h * dy_stride_h + off_m * dy_stride_m + cols2
            x_ptrs2 = X + off_b * x_stride_b + off_h * x_stride_h \
                    + (stride * off_m + rows[:, None]) * x_stride_n \
                    + cols2[None, :]
            dy2 = tl.load(dy_ptrs2).to(tl.float32)
            x2 = tl.load(x_ptrs2, mask=mask[:, None], other=0.).to(tl.float32)
            x_pe2 = x2 + pe2
            dw += tl.sum(x_pe2 * dy2[None, :], axis=1)
            dpe2 += w[:, None] * dy2[None, :]

        current_id += NUM_SMS

    dw_ptrs = DW + pid * kernel_size + rows
    dpe_ptrs = DPE + pid * kernel_size * D + rows[:, None] * D + cols[None, :]
    tl.store(dw_ptrs, dw / kernel_size, mask=mask)
    tl.store(dpe_ptrs, dpe / kernel_size, mask=mask[:, None])
    if D2 > 0:
        dpe_ptrs2 = DPE + pid * kernel_size * D + rows[:, None] * D + cols2[None, :]
        tl.store(dpe_ptrs2, dpe2 / kernel_size, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
    ],
    key=['D1', 'D2']
)
@triton.jit
def _block_compress_dx(DY, DX,
                        W, 
                        dy_stride_b, dy_stride_m, dy_stride_h, dy_stride_d,
                        dx_stride_b, dx_stride_n, dx_stride_h, dx_stride_d,
                        stride:tl.constexpr, kernel_size, num_blocks, 
                        D1:tl.constexpr, D2:tl.constexpr,
                          ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    pid_k = tl.cast(tl.program_id(2), tl.int64)

    DY += off_b * dy_stride_b + off_h * dy_stride_h
    DX += off_b * dx_stride_b + off_h * dx_stride_h

    rows = tl.arange(0, stride)
    cols = tl.arange(0, D1)
    dx = tl.zeros((stride, D1), dtype=tl.float32)
    if D2>0:
        cols2 = tl.arange(0, D2) + D1
        dx2 = tl.zeros((stride, D2), dtype=tl.float32)
    for idx in range(0, (kernel_size/stride).to(tl.int32)):
        block_idx = pid_k - idx
        if block_idx >=0 and block_idx < num_blocks:
            dy_ptrs = DY + block_idx * dy_stride_m + cols
            dy = tl.load(dy_ptrs).to(tl.float32)
            w = tl.load(W + idx*stride + rows).to(tl.float32)
            dx += dy[None, :] * w[:, None]
            if D2 > 0:
                dy_ptrs2 = DY + block_idx * dy_stride_m + cols2
                dy2 = tl.load(dy_ptrs2).to(tl.float32)
                dx2 += dy2[None, :] * w[:, None]
    dx_ptrs = DX + (pid_k * stride + rows[:, None]) * dx_stride_n + cols[None, :]
    tl.store(dx_ptrs, dx / kernel_size)
    if D2 > 0:
        dx_ptrs2 = DX + (pid_k * stride + rows[:, None]) * dx_stride_n + cols2[None, :]
        tl.store(dx_ptrs2, dx2 / kernel_size)


class _BlockCompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, pe, stride):
        B, N, H, D = x.shape
        kernel_size = len(weight)
        assert kernel_size % stride == 0
        assert math.log2(kernel_size).is_integer()
        assert N >= kernel_size
        num_blocks = (N - kernel_size) // stride + 1
        assert num_blocks > 0

        BLOCK_SIZE_N = triton.next_power_of_2(kernel_size)
        
        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()
        y = torch.empty(B, num_blocks, H, D, device=x.device, dtype=x.dtype)
        grids = (B, H, num_blocks)
        _block_compress_fwd[grids](x, weight, pe, y,
                                   *x.stride(),
                                   *y.stride(),
                                    stride, kernel_size,
                                    D, D1, D2, BLOCK_SIZE_N
                                   )
        ctx.save_for_backward(x, weight, pe)
        ctx.infos = (B, H, N, D, kernel_size, stride, num_blocks, D1, D2, BLOCK_SIZE_N)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, pe = ctx.saved_tensors
        B, H, N, D, kernel_size, stride, num_blocks, D1, D2, BLOCK_SIZE_N = ctx.infos

        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        dw = torch.empty(NUM_SMS, kernel_size, device=x.device, dtype=torch.float32)
        dpe = torch.empty(NUM_SMS, kernel_size, D, device=x.device, dtype=torch.float32)
        _block_compress_dwdpe[(NUM_SMS,)](dy, dw, dpe,
                                          x, weight, pe,
                                         *dy.stride(),
                                         *x.stride(),
                                         stride, kernel_size, num_blocks, NUM_SMS,
                                         B, H, D, 
                                         D1, D2, BLOCK_SIZE_N
                                         )
        dw = dw.sum(0).to(weight.dtype)
        dpe = dpe.sum(0).to(pe.dtype)

        K = (stride * num_blocks + (kernel_size - stride)) // stride
        dx = torch.empty_like(x)
        dx[:, :, num_blocks * stride + kernel_size - stride:] = 0
        _block_compress_dx[(B,H,K)](dy, dx,
                                         weight,
                                         *dy.stride(),
                                         *dx.stride(),
                                         stride, kernel_size, num_blocks, 
                                         D1, D2
                                         )
        return dx, dw, dpe, None

def blcok_compress(x, weight, pe, stride):
    return _BlockCompress.apply(x, weight, pe, stride)


@triton.autotune(
    configs=[
        # Short sequences (4K-8K)
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        # Medium sequences (16K-32K)
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_stages=2, num_warps=8),
        # Long sequences (64K+)
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_M': 16}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=16),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD']
)
@triton.jit
def _comp_attn_fwd_kernel(Q, K, V, O, LSE, 
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    start_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h
    LSE += off_b * lse_stride_b + off_qh * lse_stride_h

    q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=off_n[:, None] < N, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=off_n[:, None] < N, other=0.)


    # m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    # l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) + 1e-10  # Small epsilon
    acc = tl.zeros([BLOCK_SIZE_N, VD], dtype=tl.float32)

    block_idx = tl.arange(0, BLOCK_SIZE_M)
    for start_kv_idx in range(kernel_size-1, start_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):
        k = tl.load(K + block_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=block_idx[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + block_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=block_idx[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        k_idx = block_idx * stride + kernel_size - 1
        attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

        # m_ij = tl.max(attn_score, axis=1)
        # new_m_i = tl.maximum(m_i, m_ij)
        # alpha = tl.exp(m_i - new_m_i)

        # exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        # l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        # Bound the difference to prevent overflow
        m_diff = tl.maximum(m_i - new_m_i, -50.0)  # exp(-50) is very small but safe
        alpha = tl.exp(m_diff)
        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])
        # Mask out -inf scores
        exp_attn_score = tl.where(attn_score > float('-inf'), exp_attn_score, 0.0)
        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(V + block_idx[:, None] * v_stride_m + tl.arange(0, VD)[None, :], mask=block_idx[:, None] < M, other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i
        block_idx += BLOCK_SIZE_M

    # acc /= l_i[:, None]
    # lse = m_i + tl.log(l_i)
    acc = acc / tl.maximum(l_i[:, None], 1e-10)  # Prevent division by zero
    lse = m_i + tl.log(tl.maximum(l_i, 1e-10))  # Prevent log(0)


    if start_n == 0:
        acc = tl.where(off_n[:, None]>=(kernel_size-1), acc, 0)
        lse = tl.where(off_n>=(kernel_size-1), lse, 0)
    tl.store(O + off_n[:, None] * o_stride_n + tl.arange(0, VD)[None, :], acc, mask=off_n[:, None] < N)   
    tl.store(LSE + off_n * lse_stride_n, lse, mask=off_n < N)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=2, num_warps=16),
    ],
    key=['N', 'VD']
)
@triton.jit
def _comp_attn_bwd_prepro(O,DO,Delta,
                    o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                    delta_stride_b, delta_stride_h, delta_stride_n,
                    N, VD: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N

    O += off_b * o_stride_b + off_h * o_stride_h
    DO += off_b * o_stride_b + off_h * o_stride_h
    Delta += off_b * delta_stride_b + off_h * delta_stride_h
    
    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    row_mask = rows < N
    cols = tl.arange(0, VD)
    
    o = tl.load(O + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    do = tl.load(DO + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + rows, delta, mask=row_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=16),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD']
)
@triton.jit
def _comp_attn_dkv_kernel(DK, DV, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                dk_stride_b, dk_stride_m, dk_stride_h, dk_stride_d,
                dv_stride_b, dv_stride_m, dv_stride_h, dv_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
                ):
    start_m = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE_M
    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DK += off_b * dk_stride_b + off_qh * dk_stride_h 
    DV += off_b * dv_stride_b + off_qh * dv_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h

    # Load K and V blocks
    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)

    k_idx = off_m * stride + kernel_size - 1
    for start_q_idx in range(start_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
        off_n = start_q_idx + tl.arange(0, BLOCK_SIZE_N)

        q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=off_n[:, None] < N, other=0.)
        do = tl.load(DO + off_n[:, None] * do_stride_n + tl.arange(0, VD)[None, :], mask=off_n[:, None] < N, other=0.)
        lse = tl.load(Lse + off_n * lse_stride_n, mask=off_n < N, other=0.)
        delta = tl.load(Delta + off_n * lse_stride_n, mask=off_n < N, other=0.)

        attn_score = tl.dot(q, k) 
        if D2 > 0:
            q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=off_n[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        # p = tl.exp(attn_score * sm_scale - lse[:, None])
        p = tl.exp(tl.minimum(attn_score * sm_scale - lse[:, None], 80.0))  # Cap to prevent overflow
        p = tl.where(attn_score > float('-inf'), p, 0.0)
        
        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)

    tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    tl.store(DV + off_m[:, None] * dv_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    if D2 > 0:
        tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=16),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD']
)
@triton.jit
def _comp_attn_dq_kernel(DQ, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale,  kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
                ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE_N
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h
    q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1), mask=off_n[:, None] < N, other=0.)
    acc_dq = tl.zeros((BLOCK_SIZE_N, D1), dtype=tl.float32)
    do = tl.load(DO + off_n[:, None] * do_stride_n + tl.arange(0, VD), mask=off_n[:, None] < N, other=0.)
    lse = tl.load(Lse + off_n, mask=off_n < N, other=0.)
    delta = tl.load(Delta + off_n, mask=off_n < N, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, mask=off_n[:, None] < N, other=0.)
        acc_dq2 = tl.zeros((BLOCK_SIZE_N, D2), dtype=tl.float32)

    off_m = tl.arange(0, BLOCK_SIZE_M)
    for start_kv_idx in range(kernel_size-1, start_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):

        k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
        v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
        attn_score = tl.dot(q, k) 
        if D2 > 0:
            k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        k_idx = off_m * stride + kernel_size - 1
        attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale
        
        acc_dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0), acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0), acc_dq2)
        off_m += BLOCK_SIZE_M

    tl.store(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], acc_dq, mask=off_n[:, None] < N)
    if D2 > 0:
        tl.store(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, acc_dq2, mask=off_n[:, None] < N)


class _compress_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, sm_scale):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()

        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()
        if sm_scale is None:
            sm_scale = D**-0.5
        o = torch.empty(B, N, QH, VD, device=q.device, dtype=q.dtype)
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        _comp_attn_fwd_kernel[grid](q, k, v, o, lse,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride)
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.empty(B, M, QH, D1+D2, device=q.device, dtype=q.dtype)
        dv = torch.empty(B, M, QH, VD, device=q.device, dtype=q.dtype)

        delta = torch.empty_like(lse)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        _comp_attn_bwd_prepro[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD
                              )

        
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
        _comp_attn_dkv_kernel[grid](dk, dv, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *dk.stride(),
                          *dv.stride(),
                          *do.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD
                          )
        
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]), B, QH)
        _comp_attn_dq_kernel[grid](dq, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *do.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD
                          )   
        dk = dk.view(B, M, KH, -1, D1+D2).sum(3)
        dv = dv.view(B, M, KH, -1, VD).sum(3)
        return dq, dk, dv, None, None, None


def compress_attn(q, k, v, kernel_size, stride, sm_scale=None):
    return _compress_attention.apply(q, k, v, kernel_size, stride, sm_scale)


class CompressKV(torch.nn.Module):
    def __init__(self, head_dim, kernel_size, stride):
        super().__init__()
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.pe = torch.nn.Parameter(torch.randn(kernel_size, head_dim))
        self.weight = torch.nn.Parameter(torch.randn(kernel_size,))

    def forward(self, x):
        return blcok_compress(x, self.weight, self.pe, self.stride)
    
class CompressAttn(torch.nn.Module):
    def __init__(self, qk_head_dim, v_head_dim, kernel_size, stride):
        super().__init__()
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.compress_key = CompressKV(self.qk_head_dim, kernel_size, stride)
        self.compress_value = CompressKV(self.v_head_dim, kernel_size, stride)
        self.sm_scale = qk_head_dim ** -0.5

    def forward(self, q, k, v):
        cmp_k = self.compress_key(k)
        cmp_v = self.compress_value(v)
        o, lse = compress_attn(q, cmp_k, cmp_v, self.kernel_size, self.stride, self.sm_scale)
        return o, lse, cmp_k

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 256}, num_stages=2, num_warps=8),
    ],
    key=['N', 'M', 'D1', 'D2']
)
@triton.jit
def _compute_attn_probs(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    start_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_N
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    if (start_n + BLOCK_SIZE_N) < (start_m * stride + kernel_size):
        return  
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_qh = off_kh * nrep

    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_kh * p_stride_h


    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :]<M)
    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :]<M)

    k_idx = off_m * stride + kernel_size - 1
    causal_mask = off_n[:, None] >= k_idx[None, :]
    p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for h_idx in range(nrep):
        q = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=off_n[:, None] < N, other=0.)
        lse = tl.load(Lse + h_idx * lse_stride_h + off_n * lse_stride_n, mask=off_n < N, other=0.)
        attn_score = tl.dot(q, k)
        if D2 > 0:
            q2 = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=off_n[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p += tl.exp(attn_score * sm_scale - lse[:, None])
    tl.store(P + off_n[:, None] * p_stride_n + off_m[None, :] * p_stride_m, p, mask=(off_n[:, None] < N) & (off_m[None, :] < M))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 4, 'CHUNK_N': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 8, 'CHUNK_N': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 16, 'CHUNK_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'CHUNK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 8, 'CHUNK_N': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'CHUNK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'CHUNK_N': 256}, num_stages=2, num_warps=8),
    ],
    key=['N', 'M', 'BLOCK_SIZE_K']
)
@triton.jit
def _compute_select_probs(AP, SP, FInd, BInd,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_k,
                          find_stride_b, find_stride_h, find_stride_n, find_stride_k,
                          bind_stride_b, bind_stride_h, bind_stride_k, bind_stride_n,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks, top_n, return_p: tl.constexpr,
                          B, N, M, KH, 
                          BLOCK_SIZE_K: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          CHUNK_N: tl.constexpr
                            ):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_h = off_bh % KH
    off_b = off_bh // KH
    start_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N \
            + tl.program_id(2) * BLOCK_SIZE_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)


    AP += off_b * ap_stride_b + off_h * ap_stride_h
    SP += off_b * sp_stride_b + off_h * sp_stride_h
    FInd += off_b * find_stride_b + off_h * find_stride_h
    BInd += off_b * bind_stride_b + off_h * bind_stride_h

    acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_SIZE_K)

    select_start = 0
    select_end = select_size
    compress_start = stride - kernel_size 
    num_loops = (select_size + kernel_size - stride) // stride
    compress_idx = (select_idx * select_size - kernel_size) // stride + 1
    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < M)
        p = tl.load(AP + off_n[:, None] * ap_stride_n + compress_idx[None, :] * ap_stride_m, 
                    mask=(off_n[:, None] < N) & mask[None, :], other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride
        
    if return_p:
        acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None], 9999, acc_p)
        tl.store(SP + off_n[:, None] * sp_stride_n + select_idx[None, :] * sp_stride_k, 
                  acc_p, mask=(off_n[:, None] < N) & (select_idx[None, :] < num_selcct_blocks))
    tl.store(BInd + off_n * bind_stride_n + (off_n // select_size) * bind_stride_k, off_n + 1, mask=off_n < N)
    tl.store(FInd + off_n * find_stride_n, off_n // select_size, mask=off_n < N)
    acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None],
                     -1., acc_p)
    top_n = tl.minimum(top_n, (start_n + BLOCK_SIZE_N - 1) // select_size + 1)

    for i in range(1, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(BInd + off_n * bind_stride_n + max_idx * bind_stride_k, off_n + 1, mask=off_n < N)
        tl.store(FInd + off_n * find_stride_n + i * find_stride_k, max_idx, mask=off_n < N)
        acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == max_idx[:, None],
                    -1., acc_p)


@torch.inference_mode()
def select_for_fwd_bwd(q, k, lse, kernel_size, stride, select_size, top_n, sm_scale=None, return_p=False):
    B, N, QH, D = q.shape
    B2, M, KH, D2 = k.shape
    assert QH % KH == 0
    nrep = QH // KH

    if math.log2(D).is_integer():
        D1 = D
        D2 = 0
    else:
        D1 = 2**int(math.log2(D-1))
        D2 = D - D1
        assert math.log2(D2).is_integer()

    if sm_scale is None:
        sm_scale = D**-0.5

    num_selcct_blocks = triton.cdiv(N, select_size)
    top_n = min(num_selcct_blocks, top_n)

    
    probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float16)
    
    grid = lambda meta: (B*KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _compute_attn_probs[grid](q, k, lse, probs,
                        *q.stride(),
                        *k.stride(),
                        *lse.stride(),
                        *probs.stride(),
                        sm_scale, kernel_size, stride,
                        B, N, M, KH, nrep,
                        D1, D2
                        )
    BLOCK_SIZE_K = triton.next_power_of_2(num_selcct_blocks)
    select_probs = None
    if return_p:
        select_probs = torch.zeros(B, KH, N, num_selcct_blocks, device=probs.device, dtype=torch.float16)
    fwd_ind = torch.full((B, KH, N, top_n), num_selcct_blocks, dtype=torch.int32, device=probs.device)
    bwd_ind = torch.zeros(B, KH, num_selcct_blocks, N, dtype=torch.int32, device=probs.device)
    
    grid=lambda meta: (B * KH, triton.cdiv(N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_SIZE_N']))
    _compute_select_probs[grid](probs, select_probs if return_p else probs, fwd_ind, bwd_ind,
                                *probs.stride(),
                                *(select_probs.stride() if return_p else probs.stride()),
                                *fwd_ind.stride(),
                                *bwd_ind.stride(),
                                kernel_size, stride, 
                                select_size, num_selcct_blocks, top_n, return_p,
                                B, N, M, KH,
                                BLOCK_SIZE_K
                                )
    return select_probs, fwd_ind, bwd_ind

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_stages=2, num_warps=16),
    ],
    key=['N']
)
@triton.jit
def _fix_bwd_indices(Ind, Cnt,
                ind_stride_b, ind_stride_h, ind_stride_k, ind_stride_n,
                cnt_stride_b, cnt_stride_h, cnt_stride_k,
                N,
                BLOCK_SIZE_N: tl.constexpr, 
                ):
    
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_k = tl.cast(tl.program_id(2), tl.int64)

    Ind += off_b * ind_stride_b + off_h * ind_stride_h + off_k * ind_stride_k
    Cnt += off_b * cnt_stride_b + off_h * cnt_stride_h + off_k * cnt_stride_k

    last_cnt = 0
    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, N, BLOCK_SIZE_N):
        off_n = start_n + cols
        ind = tl.load(Ind + off_n, mask=off_n < N, other=0)
        this_cnt = tl.sum(ind)
        if this_cnt > 0:
            this_cnt = tl.sum(tl.where(ind == 0, 0, 1))
            ind = tl.sort(ind, descending=True)
            tl.store(Ind + last_cnt + cols, ind - 1, mask=cols < this_cnt)
            last_cnt += this_cnt
    tl.store(Cnt, last_cnt)

from copy import deepcopy

@torch.inference_mode()
def fix_bwd_ind(bwd_ind, inplace=True):
    assert bwd_ind.is_contiguous()
    if not inplace:
        bwd_ind = deepcopy(bwd_ind)
    B, KH, num_selcct_blocks, N = bwd_ind.shape
    count = torch.empty(B, KH, num_selcct_blocks, dtype=torch.int32, device=bwd_ind.device)
    _fix_bwd_indices[(B, KH, num_selcct_blocks)](bwd_ind, count,
                                                 *bwd_ind.stride(),
                                                 *count.stride(),
                                                 N
                                                 )
    return bwd_ind, count

@triton.autotune(
    configs=[
        triton.Config({'CHUNK_N': 64}, num_stages=1, num_warps=2),
        triton.Config({'CHUNK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'CHUNK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'CHUNK_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'CHUNK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'CHUNK_N': 128}, num_stages=4, num_warps=8),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD', 'BLOCK_SIZE_H', 'BLOCK_SIZE_M']
)
@triton.jit
def _select_attn_fwd_kernel(Q, 
                K, 
                V, 
                O, 
                Lse, 
                Ind,
                sm_scale, 
                top_n: tl.constexpr,
                N, 
                M, 
                KH: tl.constexpr,
                QH: tl.constexpr, 
                D1: tl.constexpr, 
                D2: tl.constexpr, 
                VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, 
                BLOCK_SIZE_M: tl.constexpr,
                CHUNK_N: tl.constexpr):
    off_n = tl.program_id(0) * CHUNK_N + tl.program_id(1)
    if off_n >= N:
        return
    off_bh = tl.program_id(2)
    off_kh = off_bh % KH
    off_b = off_bh // KH

    D = D1 + D2
    nrep = QH // KH
    strat_qh = nrep * off_kh
    
    Q += (off_b * N + off_n) * QH * D + strat_qh * D
    O += (off_b * N + off_n) * QH * VD + strat_qh * VD
    K += (off_b * M * KH + off_kh) * D
    V += (off_b * M * KH + off_kh) * VD
    Ind += (off_b * N * KH + off_n + off_kh * N) * top_n
    Lse += (off_b * N + off_n) * QH


    q_ptrs = tl.make_block_ptr(Q, (nrep, D), (D, 1), (0, 0),(BLOCK_SIZE_H, D1), (1,0))
    q = tl.load(q_ptrs, boundary_check=(0,1))
    if D2 > 0:
        q_ptrs2 = tl.make_block_ptr(Q, (nrep, D), (D, 1), (0, D1),(BLOCK_SIZE_H, D2), (1,0))
        q2 = tl.load(q_ptrs2, boundary_check=(0,1))

    m_i = tl.full([BLOCK_SIZE_H], float("-inf"), dtype=tl.float32)
    # l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) + 1e-10
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    for i in range(0, stop_n):
        start_m = tl.load(Ind + i).to(tl.int32) * BLOCK_SIZE_M
        k_ptrs = tl.make_block_ptr(K, (D, M), (1, KH * D), (0, start_m), (D1, BLOCK_SIZE_M), (0,1))
        v_ptrs = tl.make_block_ptr(V, (M, VD), (KH * VD , 1), (start_m, 0), (BLOCK_SIZE_M, VD), (1, 0))
        k = tl.load(k_ptrs, boundary_check=(0, 1))
        v = tl.load(v_ptrs, boundary_check=(0, 1))
        attn_score = tl.dot(q, k)
        if D2>0:
            k_ptrs2 = tl.make_block_ptr(K, (D, M), (1, KH * D), (D1, start_m), (D2, BLOCK_SIZE_M), (0,1))
            k2 = tl.load(k_ptrs2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        attn_score = tl.where(off_n >= (start_m + tl.arange(0, BLOCK_SIZE_M))[None, :], attn_score, float('-inf'))

        new_m_i = tl.maximum(m_i, tl.max(attn_score, axis=1))
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)

        acc = acc * alpha[:, None] + tl.dot(exp_attn_score.to(v.dtype), v)
        m_i = new_m_i


    acc /= l_i[:, None]
    o_ptrs = tl.make_block_ptr(O, (nrep, VD), (VD, 1), (0, 0),(BLOCK_SIZE_H, VD), (1,0))
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), boundary_check=(0,1))
    lse = m_i + tl.log(l_i)
    tl.store(Lse + strat_qh + tl.arange(0, BLOCK_SIZE_H), lse, mask=tl.arange(0, BLOCK_SIZE_H) < nrep)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['N', 'VD']
)
@triton.jit
def _select_attn_bwd_prepo_kernel(O,DO,Delta,
                    o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                    delta_stride_b, delta_stride_n, delta_stride_h,
                    N, VD: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N

    O += off_b * o_stride_b + off_h * o_stride_h
    DO += off_b * o_stride_b + off_h * o_stride_h
    Delta += off_b * delta_stride_b + off_h * delta_stride_h
    
    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    row_mask = rows < N
    cols = tl.arange(0, VD)
    
    o = tl.load(O + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    do = tl.load(DO + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + rows * delta_stride_n, delta, mask=row_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=2, num_warps=8),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD', 'BLOCK_SIZE_M']
)
@triton.jit
def _select_attn_bwd_dkv_kernel(DK, DV, DO, 
                Q, K, V, 
                Lse, Delta, Ind, Count,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                dk_stride_b, dk_stride_m, dk_stride_h, dk_stride_d,
                dv_stride_b, dv_stride_m, dv_stride_h, dv_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_n, lse_stride_h,
                ind_stride_b, ind_stride_h, ind_stride_m, ind_stride_n,
                cnt_stride_b, cnt_stride_h, cnt_stride_m,
                sm_scale, 
                N, M, nrep,  
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr,
                ):
    pid0 = tl.cast(tl.program_id(0), tl.int64) 
    off_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    off_kh = off_qh // nrep

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    DK += off_b * dk_stride_b + off_qh * dk_stride_h
    DV += off_b * dv_stride_b + off_qh * dv_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h 
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + pid0 * ind_stride_m
    Count += off_b * cnt_stride_b + off_kh * cnt_stride_h

    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)

    count = tl.load(Count + pid0)
    for start in range(0, count, BLOCK_SIZE_N):
        off_ind = start + tl.arange(0, BLOCK_SIZE_N)
        q_idx = tl.load(Ind + off_ind, off_ind < count, other=0)
        q_idx = tl.where(off_ind < count, q_idx, N)
        q = tl.load(Q + q_idx[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=q_idx[:, None] < N, other=0.)
        lse = tl.load(Lse + q_idx * lse_stride_n, mask=q_idx < N, other=0.)
        attn_score = tl.dot(q, k)
        if D2 > 0:
            q2 = tl.load(Q + q_idx[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=q_idx[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(q_idx[:, None] >= off_m[None, :], attn_score, float('-inf'))
        
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        do = tl.load(DO + q_idx[:, None] * do_stride_n + tl.arange(0, VD)[None, :], mask=q_idx[:, None] < N, other=0.)
        
        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)
        
        
        delta = tl.load(Delta + q_idx * lse_stride_n, mask=q_idx < N, other=0.)
        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale
        acc_dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
    
    tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    tl.store(DV + off_m[:, None] * dv_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    if D2 > 0:
        tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)

@triton.autotune(
    configs=[
        triton.Config({'CHUNK_N': 64}, num_warps=2, num_stages=1),
        triton.Config({'CHUNK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'CHUNK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'CHUNK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'CHUNK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'CHUNK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['N', 'M', 'D1', 'D2', 'VD', 'BLOCK_SIZE_H', 'BLOCK_SIZE_M']
)
@triton.jit
def _select_attn_bwd_dq_kernel( DQ, 
                DO, 
                Q, 
                K, 
                V, 
                Lse, 
                Delta, 
                Ind, 
                sm_scale, 
                top_n,
                N: tl.constexpr, 
                M: tl.constexpr, 
                QH: tl.constexpr,
                KH: tl.constexpr, 
                D1: tl.constexpr, 
                D2: tl.constexpr, 
                VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, 
                BLOCK_SIZE_M: tl.constexpr,
                CHUNK_N: tl.constexpr,
                ):
    off_n = tl.program_id(0) * CHUNK_N + tl.program_id(1)
    if off_n >= N:
        return
    off_bh = tl.program_id(2)
    off_kh = off_bh % KH
    off_b = off_bh // KH

    D = D1 + D2
    nrep = QH // KH
    off_qh = off_kh * nrep

    Q += (off_b * N * QH + off_n * QH + off_qh) * D
    K += (off_b * M * KH + off_kh) * D
    V += (off_b * M * KH + off_kh) * VD
    DQ += (off_b * N * QH + off_n * QH + off_qh) * D
    DO += (off_b * N * QH + off_n * QH + off_qh) * VD
    Lse += (off_b * N + off_n) * QH + off_qh
    Delta += (off_b * N + off_n) * QH + off_qh
    Ind += (off_b * N * KH + off_n + off_kh * N) * top_n

    
    q_ptrs = tl.make_block_ptr(Q, (nrep, D), (D, 1), (0, 0),(BLOCK_SIZE_H, D1), (1,0))
    do_ptrs = tl.make_block_ptr(DO, (nrep, VD), (VD, 1), (0, 0),(BLOCK_SIZE_H, VD), (1,0))
    q = tl.load(q_ptrs, boundary_check=(0,1))
    do = tl.load(do_ptrs, boundary_check=(0, 1))
    heads = tl.arange(0, BLOCK_SIZE_H)
    lse = tl.load(Lse + heads, mask=heads<nrep, other=0.)
    delta = tl.load(Delta + heads, mask=heads<nrep, other=0.)
    acc_dq = tl.zeros([BLOCK_SIZE_H, D1], dtype=tl.float32)

    if D2 > 0:
        q_ptrs2 = tl.make_block_ptr(Q, (nrep, D), (D, 1), (0, D1),(BLOCK_SIZE_H, D2), (1,0))
        q2 = tl.load(q_ptrs2, boundary_check=(0,1))
        acc_dq2 = tl.zeros([BLOCK_SIZE_H, D2], dtype=tl.float32)

    
    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    for i in range(0, stop_n):
        select_idx = tl.load(Ind + i)
        start_m = select_idx * BLOCK_SIZE_M
        k_ptrs = tl.make_block_ptr(K, (D, M), (1, KH * D), (0, start_m), (D1, BLOCK_SIZE_M), (0, 1))
        v_ptrs = tl.make_block_ptr(V, (VD, M), (1, KH * VD), (0, start_m), (VD, BLOCK_SIZE_M), (0, 1))
        k = tl.load(k_ptrs, boundary_check=(0,1))
        attn_score = tl.dot(q, k)
        if D2>0:
            k_ptrs2 = tl.make_block_ptr(K, (D, M), (1, KH * D), (D1, start_m), (D2, BLOCK_SIZE_M), (0, 1))
            k2 = tl.load(k_ptrs2, boundary_check=(0,1))
            attn_score = tl.dot(q2, k2, attn_score)
        v = tl.load(v_ptrs, boundary_check=(0,1))
        dp = tl.dot(do, v)
        
        attn_score = tl.where(off_n >= (start_m + tl.arange(0, BLOCK_SIZE_M))[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
    
        ds = p * (dp - delta[:, None]) * sm_scale

        acc_dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0), acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0), acc_dq2)
    
    dq_ptrs = tl.make_block_ptr(DQ, (nrep, D), (D, 1), (0, 0),(BLOCK_SIZE_H, D1), (1,0))
    tl.store(dq_ptrs, acc_dq.to(dq_ptrs.dtype.element_ty), boundary_check=(0, 1))
    if D2 > 0:
        dq_ptrs2 = tl.make_block_ptr(DQ, (nrep, D), (D, 1), (0, D1),(BLOCK_SIZE_H, D2), (1,0))
        tl.store(dq_ptrs2, acc_dq2.to(dq_ptrs.dtype.element_ty), boundary_check=(0, 1))

class _select_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, select_size, fwd_ind, bwd_ind, sm_scale, inplace):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()
        assert math.log2(select_size).is_integer()
        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()

        if sm_scale is None:
            sm_scale = D**-0.5

        o = torch.empty(B, N, QH, VD, device=q.device, dtype=q.dtype)
        lse = torch.empty(B, N,QH, dtype=torch.float32, device=q.device,)

        nrep = QH // KH
        BLOCK_SIZE_H = max(triton.next_power_of_2(nrep), 16)
        BLOCK_SIZE_M = select_size
        top_n = fwd_ind.size(-1)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'], B * KH)
        _select_attn_fwd_kernel[grid](q, k, v, o, lse, fwd_ind,
                        sm_scale, top_n,
                        N, M, KH, QH,
                        D1, D2, VD,
                        BLOCK_SIZE_H, BLOCK_SIZE_M
                        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.bwd_ind = bwd_ind
        ctx.fwd_ind = fwd_ind
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, top_n, BLOCK_SIZE_H, BLOCK_SIZE_M)
        ctx.inplace = inplace
        return o

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        bwd_ind, count = fix_bwd_ind(ctx.bwd_ind, ctx.inplace)
        B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, top_n, BLOCK_SIZE_H, BLOCK_SIZE_M = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        

        delta = torch.empty_like(lse)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        _select_attn_bwd_prepo_kernel[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD
                              )
    
        
        dk = torch.empty(B, M, QH, D1+D2, device=k.device, dtype=k.dtype)
        dv = torch.empty(B, M, QH, VD, device=k.device, dtype=k.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
        _select_attn_bwd_dkv_kernel[grid](dk, dv, do, 
                          q, k, v,
                          lse, delta, 
                          bwd_ind,
                          count,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *dk.stride(),
                          *dv.stride(),
                          *do.stride(),
                          *lse.stride(),
                          *bwd_ind.stride(),
                          *count.stride(),
                          sm_scale, 
                          N, M, nrep,  
                          D1, D2, VD,
                          BLOCK_SIZE_M
                          )
        dk = dk.view(B, M, KH, nrep, -1).sum(3)
        dv = dv.view(B, M, KH, nrep, -1).sum(3)
        
        dq = torch.empty_like(q)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'], B * KH)
        _select_attn_bwd_dq_kernel[grid](dq, do, 
                          q, k, v,
                          lse, delta, 
                          ctx.fwd_ind,
                          sm_scale, top_n,
                          N, M, QH,KH,  
                          D1, D2, VD,
                          BLOCK_SIZE_H, BLOCK_SIZE_M
                          )
        return dq, dk, dv, None, None, None, None, None


def select_attn(q, k, v, select_size, fwd_ind, bwd_ind, sm_scale=None, inplace=True):
    return _select_attention.apply(q, k, v, select_size, fwd_ind, bwd_ind, sm_scale, inplace)

class NsaAttention(torch.nn.Module):
    """
    native sparse attention.

    Args:
        qk_head_dim (int): head dim of q and k head

        v_head_dim (int): head dim of v head

        kernel_size (int): how many kv will be compressed and become a compressed kv block, the "l" in the paper

        stride (int): like conv stride, compress the next block will move how many kv, the "d" in the paper

        select_size (int): select block size, the "l'" in the paper

        top_n (int): q will chosses how many select blocks.

        window_size (int): sliding window size for window attention
    """
    def __init__(self, qk_head_dim, v_head_dim, kernel_size=32, stride=16, select_size=64, top_n=16, window_size=512):
        super().__init__()
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.select_size = select_size
        self.top_n = top_n
        self.window_size = window_size
        self.sm_scale = qk_head_dim ** -0.5
        assert math.log2(self.stride).is_integer()
        assert kernel_size % stride == 0 and select_size % kernel_size == 0

        self.compress_attn = CompressAttn(qk_head_dim, v_head_dim, kernel_size, stride)
        self.select_for_fwd_bwd = partial(select_for_fwd_bwd, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      select_size=self.select_size, 
                                      top_n=self.top_n, 
                                      sm_scale=self.sm_scale)
        
        self.select_attn = partial(select_attn, 
                                   select_size=self.select_size, 
                                   sm_scale=self.sm_scale)
        
        self.window_attn = partial(flash_attn_func, 
                                   softmax_scale=self.sm_scale, 
                                   causal=True, 
                                   window_size=(self.window_size, -1) )
        self.attn_gate = torch.nn.Linear(qk_head_dim, 3)

    
    def forward(self, q, k, v, inplace=True):
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [b, seq_len, num_q_head, qk_head_dim]
            k (torch.Tensor): [b, seq_len, num_kv_head, qk_head_dim]
            v (torch.Tensor): [b, seq_len, num_kv_head, v_head_dim]
            inplace (bool): in the backward the bwd_ind will be update in-place, set False for benchmark
        Returns:
            o (torch.Tensor): [b, seq_len, num_q_head, v_head_dim]
        """
        cmp_o, lse, cmp_k = self.compress_attn(q, k, v)
        _, fwd_ind, bwd_ind = self.select_for_fwd_bwd(q, cmp_k, lse)
        select_o = self.select_attn(q, k, v, fwd_ind=fwd_ind, bwd_ind=bwd_ind, inplace=inplace)
        window_o = self.window_attn(q, k, v)
        if isinstance(window_o, Tuple):
            window_o = window_o[0]
        weight = self.attn_gate(q)
        combine_o = fused_attention(cmp_o, select_o, window_o, weight)
        return combine_o




####################################################################################################################################################
import math
import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}

result_gold = {}

class NsaAttention(torch.nn.Module):
    """
    native sparse attention.

    Args:
        qk_head_dim (int): head dim of q and k head

        v_head_dim (int): head dim of v head

        kernel_size (int): how many kv will be compressed and become a compressed kv block, the "l" in the paper

        stride (int): like conv stride, compress the next block will move how many kv, the "d" in the paper

        select_size (int): select block size, the "l'" in the paper

        top_n (int): q will chosses how many select blocks.

        window_size (int): sliding window size for window attention
    """
    def __init__(self, qk_head_dim, v_head_dim, kernel_size=32, stride=16, select_size=64, top_n=16, window_size=512):
        super().__init__()
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.select_size = select_size
        self.top_n = top_n
        self.window_size = window_size
        self.sm_scale = qk_head_dim ** -0.5
        assert math.log2(self.stride).is_integer()
        assert kernel_size % stride == 0 and select_size % kernel_size == 0

        self.compress_attn = CompressAttn(qk_head_dim, v_head_dim, kernel_size, stride)
        self.select_for_fwd_bwd = partial(select_for_fwd_bwd, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      select_size=self.select_size, 
                                      top_n=self.top_n, 
                                      sm_scale=self.sm_scale)
        
        self.select_attn = partial(select_attn, 
                                   select_size=self.select_size, 
                                   sm_scale=self.sm_scale)
        
        self.window_attn = partial(flash_attn_func, 
                                   softmax_scale=self.sm_scale, 
                                   causal=True, 
                                   window_size=(self.window_size, -1) )
        self.attn_gate = torch.nn.Linear(qk_head_dim, 3)

    
    def forward(self, q, k, v, inplace=True):
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [b, seq_len, num_q_head, qk_head_dim]
            k (torch.Tensor): [b, seq_len, num_kv_head, qk_head_dim]
            v (torch.Tensor): [b, seq_len, num_kv_head, v_head_dim]
            inplace (bool): in the backward the bwd_ind will be update in-place, set False for benchmark
        Returns:
            o (torch.Tensor): [b, seq_len, num_q_head, v_head_dim]
        """
        cmp_o, lse, cmp_k = self.compress_attn(q, k, v)
        _, fwd_ind, bwd_ind = self.select_for_fwd_bwd(q, cmp_k, lse)
        select_o = self.select_attn(q, k, v, fwd_ind=fwd_ind, bwd_ind=bwd_ind, inplace=inplace)
        window_o = self.window_attn(q, k, v)
        if isinstance(window_o, Tuple):
            window_o = window_o[0]
        weight = self.attn_gate(q)
        combine_o = fused_attention(cmp_o, select_o, window_o, weight)
        return combine_o
    
######################################## HELPERS for Eval ######################################## 
# Helper function to define GB/s for NSA
def calculate_nsa_gbps(params: Dict, ms: float) -> float:
    b = params['b']
    n = params['n']
    qh = params['qh']
    kh = params['kh']
    d = params['d']
    vd = params['vd']
    dtype = dtype_mapping[params['dtype_str']]
    
    # Calculate bytes for NSA operations
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Read Q, K, V, write O
    total_bytes = (b * n * qh * d +  # Q
                   b * n * kh * d +   # K
                   b * n * kh * vd +  # V
                   b * n * qh * vd) * bytes_per_element  # O
    
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for NSA
def calculate_nsa_tflops(params: Dict, ms: float) -> float:
    b = params['b']
    n = params['n']
    qh = params['qh']
    kh = params['kh']
    d = params['d']
    vd = params['vd']
    
    # Approximate FLOPs for attention operations
    # QK^T: 2 * b * qh * n * n * d
    # Softmax: ~5 * b * qh * n * n (approximation)
    # Attention * V: 2 * b * qh * n * n * vd
    flops = 2 * b * qh * n * n * d + 5 * b * qh * n * n + 2 * b * qh * n * n * vd
    
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ######################################## 


OP_NAME_FOR_BENCHMARK = "nsa_attention_perf"

# (b, n, qh, kh, d, vd, stride, kernel_size, select_size, top_n, window_size, dtype_str)
# Test parameters for NSA
test_params = [
    (1, 4096, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
    (1, 8192, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
    (1, 16384, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
    (1, 32768, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
    (1, 65536, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
    (1, 131072, 64, 4, 128, 128, 16, 32, 64, 16, 512, 'bfloat16'),
]
@pytest.mark.parametrize('b,n,qh,kh,d,vd,stride,kernel_size,select_size,top_n,window_size,dtype_str', test_params)
def test_nsa(b, n, qh, kh, d, vd, stride, kernel_size, select_size, top_n, window_size, dtype_str, request):
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    device = 'cuda'
    
    # Create inputs
    q = torch.randn(b, n, qh, d, device=device, dtype=dtype)
    k = torch.randn(b, n, kh, d, device=device, dtype=dtype)
    v = torch.randn(b, n, kh, vd, device=device, dtype=dtype)
    
    # Initialize NSA module
    nsa = NsaAttention(d, vd, kernel_size, stride, select_size, top_n, window_size).to(device).to(dtype)
    
    # Forward pass
    output = nsa(q, k, v, inplace=False)
    
    # torch.set_printoptions(profile='full')
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    ################################################################### 
    
    # Basic shape check
    assert output.shape == (b, n, qh, vd)


@pytest.mark.parametrize('b,n,qh,kh,d,vd,stride,kernel_size,select_size,top_n,window_size,dtype_str', test_params)
def test_performance(b, n, qh, kh, d, vd, stride, kernel_size, select_size, top_n, window_size, dtype_str, request):
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    device = 'cuda'
    
    # Create inputs
    q = torch.randn(b, n, qh, d, device=device, dtype=dtype)
    k = torch.randn(b, n, kh, d, device=device, dtype=dtype)
    v = torch.randn(b, n, kh, vd, device=device, dtype=dtype)
    
    # Initialize NSA module
    nsa = NsaAttention(d, vd, kernel_size, stride, select_size, top_n, window_size).to(device).to(dtype)
    
    # Operation lambda
    op_lambda = lambda: nsa(q, k, v, inplace=False)
    

    bench_config = do_bench_config(
        warm_up=10,      
        repetition=50,   
        quantiles=[0.5, 0.2, 0.8] 
    )
    
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    
    # Parameters for calculators
    current_params_for_calculators = {
        "b": b, "n": n, "qh": qh, "kh": kh, "d": d, "vd": vd,
        "stride": stride, "kernel_size": kernel_size, 
        "select_size": select_size, "top_n": top_n,
        "window_size": window_size, "dtype_str": dtype_str
    }
    
    benchmarker.run_benchmark(current_params_dict=current_params_for_calculators,
                              gbps_calculator=calculate_nsa_gbps,
                              tflops_calculator=calculate_nsa_tflops)

######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    # import pdb; pdb.set_trace()
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################



                










