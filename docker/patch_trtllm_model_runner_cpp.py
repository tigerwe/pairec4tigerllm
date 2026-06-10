#!/usr/bin/env python3
"""Patch TensorRT-LLM ModelRunnerCpp.from_dir to accept scheduler_config.

TensorRT-LLM 1.0.0 in the validated ARM runtime builds ExecutorConfig inside
ModelRunnerCpp.from_dir but does not expose scheduler_config in that wrapper.
The project needs it to keep C++ KV cache pressure diagnostics deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_line(lines: list[str], start: int, needle) -> int:
    for i in range(start, len(lines)):
        if needle(lines[i]):
            return i
    raise RuntimeError("target line not found")


def patch_model_runner(path: Path) -> bool:
    lines = path.read_text().splitlines(True)
    changed = False

    def_i = _find_line(lines, 0, lambda x: "def from_dir(" in x)
    header_end = _find_line(
        lines,
        def_i,
        lambda x: x.lstrip().startswith(")") or x.rstrip().endswith("):"),
    )
    header = "".join(lines[def_i : header_end + 1])
    if "scheduler_config" not in header:
        indent = " " * (len(lines[header_end]) - len(lines[header_end].lstrip()))
        lines.insert(
            header_end,
            f"{indent}scheduler_config: Optional[trtllm.SchedulerConfig] = None,\n",
        )
        changed = True

    text = "".join(lines)
    lines = text.splitlines(True)

    exec_i = _find_line(lines, 0, lambda x: "trtllm.ExecutorConfig(" in x)
    exec_end = _find_line(
        lines,
        exec_i,
        lambda x: x.startswith("            )") or x.startswith("        )"),
    )
    exec_block = "".join(lines[exec_i : exec_end + 1])
    if "scheduler_config" not in exec_block:
        insert_i = _find_line(lines, exec_i, lambda x: "max_beam_width=max_beam_width" in x)
        indent = " " * (len(lines[insert_i]) - len(lines[insert_i].lstrip()))
        lines.insert(insert_i + 1, f"{indent}scheduler_config=scheduler_config,\n")
        changed = True

    if changed:
        path.write_text("".join(lines))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args()

    changed = patch_model_runner(args.path)
    print(f"patched={changed} path={args.path}")


if __name__ == "__main__":
    main()
