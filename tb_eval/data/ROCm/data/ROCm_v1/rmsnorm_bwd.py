# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
####################### Imports #####################
import argparse
import torch
import sys
import pytest
from itertools import product

import triton
import triton.language as tl
####################### Imports #####################



@triton.jit
def rms_bwd_kernel(grad_output_ptr, input_ptr, g_ptr, rsigma_ptr, dx_ptr, dg_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                   USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    #   tl.assume(input_row_stride >= 0)
    #   tl.assume(output_row_stride >= 0)
    #   tl.assume(row_start >= 0)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * input_row_stride

            # Compute gradients sum of all colums for each row
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # comment out for now to make it compatible with triton 3.1
            # grad_sum: tl.float32 = 0.0
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1.
                grad_sum += tl.sum(grad_output * x * g, axis=0)

            # remainder for grad_sum:
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.
            grad_sum += tl.sum(grad_output * x * g, axis=0)

            # Load r_sigma
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)

                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1.
                grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                              n_cols)

                dx_ptrs = row_dx_ptr + cols
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))

                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                tl.store(dg_ptrs, dg.to(tl.float32))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols

            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.
            grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                          n_cols)

            dx_ptrs = row_dx_ptr + cols
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            grad_output_ptrs = grad_output_ptr + row_idx * output_row_stride + col_offsets
            dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets
            dg_ptrs = dg_ptr + row_idx * input_row_stride + col_offsets

            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16, ))
            dx_ptrs = tl.multiple_of(dx_ptrs, (16, ))

            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1.

            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            grad_sum = tl.sum(grad_output * x * g, axis=0)

            grad_input = grad_output * norm_factor * g - (norm_factor * norm_factor * norm_factor) * x * (grad_sum /
                                                                                                          n_cols)
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)





