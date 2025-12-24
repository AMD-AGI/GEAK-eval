#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict

TB_PREFIX = "TritonBench_v1."
DEEP_PREFIX = "geak_eval.data.TritonBench.data.TritonBench_G_v1."
PERF_BASELINE_MOD = "performance_utils"
PERF_REFACTORED_MOD = "geak_eval.perf.performance_utils"
PERF_REFACTORED_LINE = "from geak_eval.perf.performance_utils import Performance_Metrics, do_bench_config\n"

CALL_OP_NAME = "call_op"
CALL_OP_REF_NAME = "call_op_ref"


class ImportToRef:
    def __init__(self, submodule: str, name: str, asname: Optional[str]):
        self.submodule = submodule
        self.name = name
        self.asname = asname

    @property
    def used_name(self) -> str:
        return self.asname or self.name

    @property
    def deep_module(self) -> str:
        return f"{DEEP_PREFIX}{self.submodule}"

    @property
    def deep_import_line(self) -> str:
        return f"from {self.deep_module} import {self.name} as {self.name}_ref\n"


def parse_tritonbench_imports(tree: ast.AST) -> Tuple[List[Tuple[int, int]], List[ImportToRef]]:
    ranges: List[Tuple[int, int]] = []
    collected: List[ImportToRef] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and hasattr(node, 'end_lineno'):
            if node.module.startswith(TB_PREFIX):
                submodule = node.module[len(TB_PREFIX):]
                ranges.append((node.lineno, node.end_lineno))
                for alias in node.names:
                    collected.append(ImportToRef(submodule, alias.name, alias.asname))
    return ranges, collected


def parse_performance_utils_imports(tree: ast.AST) -> Tuple[List[Tuple[int, int]], bool]:
    ranges: List[Tuple[int, int]] = []
    already_has_ref = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and hasattr(node, 'end_lineno'):
            if node.module == PERF_BASELINE_MOD:
                ranges.append((node.lineno, node.end_lineno))
            if node.module == PERF_REFACTORED_MOD:
                already_has_ref = True
    return ranges, already_has_ref


def find_import_block_end(tree: ast.AST) -> int:
    last = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(node, 'end_lineno'):
            last = max(last, node.end_lineno)
    return last


def find_function_range(tree: ast.AST, func_name: str) -> Optional[Tuple[int, int]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node.lineno, node.end_lineno
    return None


def already_has_call_op_ref(tree: ast.AST) -> bool:
    return find_function_range(tree, CALL_OP_REF_NAME) is not None


def insert_lines_at(lines: List[str], after_line: int, new_lines: List[str]) -> List[str]:
    idx = min(max(after_line, 0), len(lines))
    return lines[:idx] + new_lines + lines[idx:]


def clone_call_op_to_call_op_ref(src_lines: List[str],
                                 call_op_range: Tuple[int, int],
                                 tb_imports: List[ImportToRef]) -> str:
    start, end = call_op_range
    block = "".join(src_lines[start - 1:end])
    block = re.sub(rf"(\bdef\s+){CALL_OP_NAME}(\s*\()", rf"\1{CALL_OP_REF_NAME}\2", block, count=1)

    for imp in tb_imports:
        for cand in {imp.used_name, imp.name}:
            if not cand:
                continue
            pattern = rf"(\b){re.escape(cand)}(\s*\()"
            repl = rf"\1{imp.name}_ref\2"
            block = re.sub(pattern, repl, block)

    if not block.endswith("\n"):
        block += "\n"
    return "\n" + block


def process_text(text: str, path: Path) -> Tuple[str, Dict[str, object]]:
    report = {
        "rewrote_tb_imports": 0,
        "added_deep_imports": 0,
        "replaced_perf_utils": 0,
        "added_perf_utils": False,
        "added_call_op_ref": False,
        "added_stub": False,
    }

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, report

    lines = text.splitlines(keepends=True)

    # --- Handle TritonBench_v1 imports ---
    tb_ranges, tb_imports = parse_tritonbench_imports(tree)
    if tb_ranges:
        report["rewrote_tb_imports"] = len(tb_ranges)
        tb_line_nos = {i for start, end in tb_ranges for i in range(start, end + 1)}
        new_lines = []
        for i, line in enumerate(lines, start=1):
            if i in tb_line_nos and line.strip().startswith("from TritonBench_v1."):
                new_lines.append(line.replace("from TritonBench_v1.", "from ", 1))
            else:
                new_lines.append(line)
        lines = new_lines

    text_after_tb = "".join(lines)
    tree_after_tb = ast.parse(text_after_tb)

    # --- Replace performance_utils import ---
    perf_ranges, already_has_ref = parse_performance_utils_imports(tree_after_tb)
    if perf_ranges:
        report["replaced_perf_utils"] = len(perf_ranges)
        perf_line_nos = {i for start, end in perf_ranges for i in range(start, end + 1)}
        new_lines = []
        for i, line in enumerate(lines, start=1):
            if i in perf_line_nos:
                continue
            new_lines.append(line)
        lines = new_lines

    text_after_removals = "".join(lines)
    tree_after_removals = ast.parse(text_after_removals)
    import_block_end = find_import_block_end(tree_after_removals)

    # --- Add new imports ---
    new_imports: List[str] = []
    seen: Set[str] = set()
    for imp in tb_imports:
        line = imp.deep_import_line
        if line not in seen:
            new_imports.append(line)
            seen.add(line)
    report["added_deep_imports"] = len(new_imports)

    if not already_has_ref:
        if PERF_REFACTORED_LINE not in text_after_removals:
            new_imports.append(PERF_REFACTORED_LINE)
            report["added_perf_utils"] = True

    if new_imports:
        lines = insert_lines_at(lines, import_block_end, new_imports)

    # --- Add call_op_ref ---
    text_after_imports = "".join(lines)
    tree_after_imports = ast.parse(text_after_imports)
    if not already_has_call_op_ref(tree_after_imports):
        rng = find_function_range(tree_after_imports, CALL_OP_NAME)
        if rng:
            cloned = clone_call_op_to_call_op_ref(text_after_imports.splitlines(keepends=True), rng, tb_imports)
            lines = insert_lines_at(lines, rng[1], [cloned])
            report["added_call_op_ref"] = True
        else:
            stub = (
                "\n\ndef call_op_ref(*args, **kwargs):\n"
                "    raise NotImplementedError(\n"
                "        'Auto-generated: call_op not found to clone; please implement call_op_ref manually.'\n"
                "    )\n"
            )
            lines.append(stub)
            report["added_stub"] = True

    return "".join(lines), report


def should_process_file(path: Path) -> bool:
    return path.suffix == ".py" and path.name != "__init__.py"


def refactor_file(src: Path, dst: Optional[Path], in_place: bool = False, dry_run: bool = False) -> None:
    text = src.read_text(encoding="utf-8")
    new_text, report = process_text(text, src)
    changed = (new_text != text)
    header = f"[{src}]"

    if dry_run:
        print(header, "DRY-RUN", "CHANGED" if changed else "UNCHANGED", report)
        return

    if in_place:
        if changed:
            src.write_text(new_text, encoding="utf-8")
            print(header, "UPDATED", report)
        else:
            print(header, "UNCHANGED")
    else:
        assert dst is not None
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(new_text, encoding="utf-8")
        print(header, "WROTE ->", dst, report)


def main():
    ap = argparse.ArgumentParser(description="Refactor TritonBench baseline files.")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.in_place and args.out_dir:
        ap.error("Use either --in-place or --out-dir, not both.")
    if not args.in_place and not args.out_dir:
        ap.error("Either --in-place or --out-dir is required.")

    for src in args.in_dir.rglob("*.py"):
        if not should_process_file(src):
            continue
        if args.in_place:
            refactor_file(src, None, in_place=True, dry_run=args.dry_run)
        else:
            rel = src.relative_to(args.in_dir)
            dst = args.out_dir / rel
            refactor_file(src, dst, in_place=False, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
