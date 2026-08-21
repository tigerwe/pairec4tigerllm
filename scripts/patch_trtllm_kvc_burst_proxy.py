#!/usr/bin/env python3
"""Patch TensorRT-LLM DataSystem Get paths with the PaiRec KVC burst proxy."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "PAIREC_KVC_BURST_PROXY_V4"
LEGACY_MARKERS = ("PAIREC_KVC_BURST_PROXY_V3", "PAIREC_KVC_BURST_PROXY_V2")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor in {path}, found {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1))


def write_managed(path: Path, content: str) -> None:
    managed = f"/* {MARKER} */\n{content}"
    if path.exists():
        existing = path.read_text()
        markers = (MARKER, *LEGACY_MARKERS)
        if not any(marker in existing for marker in markers):
            raise RuntimeError(f"refusing to overwrite unmanaged file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(managed)


def managed_sources(repo_root: Path) -> tuple[str, str, str]:
    source = repo_root / "cpp/kvc_burst"
    shared = (source / "kvc_burst_shared.h").read_text()
    proxy_header = (source / "kvc_operation_proxy.h").read_text().replace(
        '#include "kvc_burst_shared.h"',
        '#include "tensorrt_llm/batch_manager/kvcBurstShared.h"')
    proxy_source = (source / "kvc_operation_proxy.cpp").read_text().replace(
        '#include "kvc_operation_proxy.h"',
        '#include "tensorrt_llm/batch_manager/kvcOperationProxy.h"')
    return shared, proxy_header, proxy_source


def patch_wrapped_get(transfer: Path, attribution_prefix: str, proxy_prefix: str,
    api: str, key_count: str, calls: tuple[str, ...]) -> None:
    text = transfer.read_text()
    if f"auto const {proxy_prefix}Token =" in text:
        return
    attribution = f"auto const {attribution_prefix}Token = beginDataSystemOperation(DataSystemOperation::kGet);"
    if text.count(attribution) != 1:
        raise RuntimeError(f"missing exact attribution anchor for {proxy_prefix} in {transfer}")
    begin = (
        f"auto const {proxy_prefix}RequestId = currentDataSystemRequestId();\n"
        f"            auto const {proxy_prefix}Token = pairec::kvc_burst::beginBusinessGet(\n"
        f"                {proxy_prefix}RequestId.value_or(\"\"), "
        f"pairec::kvc_burst::BusinessApi::{api}, {key_count});\n")
    if api == "kGet":
        begin += (
            f"            auto {proxy_prefix}Pressure = "
            "pairec::kvc_burst::beginInProcessPressure(\n"
            f"                {proxy_prefix}Token, [&kvClient1](uint32_t, std::string const& pressureKey) {{\n"
            "                    datasystem::Optional<datasystem::Buffer> pressureBuffer;\n"
            "                    auto const pressureStatus = kvClient1->Get(pressureKey, pressureBuffer, 0);\n"
            "                    return !pressureStatus.IsError() && static_cast<bool>(pressureBuffer);\n"
            "                });\n")
    begin += f"            {attribution}"
    replace_once(transfer, attribution, begin)

    text = transfer.read_text()
    matches = [call for call in calls if text.count(call) == 1]
    if len(matches) != 1:
        raise RuntimeError(f"expected one DataSystem Get call for {proxy_prefix} in {transfer}, found {len(matches)}")
    call = matches[0]
    replace_once(transfer, call,
        f"{call}\n"
        f"            auto const {proxy_prefix}EndedNs = pairec::kvc_burst::MonotonicNs();")
    attribution_finish = (
        f"finishDataSystemOperation({attribution_prefix}Token, DataSystemOperation::kGet, "
        f"{attribution_prefix}Us, getRet.IsError());")
    finish = attribution_finish
    if api == "kGet":
        finish += f"\n            {proxy_prefix}Pressure.finish();"
    finish += (f"\n            pairec::kvc_burst::finishBusinessGet(\n"
               f"                {proxy_prefix}Token, !getRet.IsError(), {proxy_prefix}EndedNs);")
    replace_once(transfer, attribution_finish, finish)


def patch_parallel_get(transfer: Path) -> None:
    text = transfer.read_text()
    if "TRTLLM_DATASYSTEM_PARALLEL_GET" not in text or "kvcBurstParallelGetToken" in text:
        return
    loop = "                for (auto const& key : keys)\n"
    begin = (
        "                auto const kvcBurstParallelGetRequestId = currentDataSystemRequestId();\n"
        "                auto const kvcBurstParallelGetToken = pairec::kvc_burst::beginBusinessGet(\n"
        "                    kvcBurstParallelGetRequestId.value_or(\"\"),\n"
        "                    pairec::kvc_burst::BusinessApi::kParallelGet, keys.size());\n"
        + loop)
    replace_once(transfer, loop, begin)
    completion = (
        "                activeParallelGetCallsAfter = "
        "gActiveParallelGetCalls.fetch_sub(1, std::memory_order_relaxed) - 1;\n")
    replace_once(transfer, completion,
        completion
        + "                pairec::kvc_burst::finishBusinessGet(kvcBurstParallelGetToken, !getFailed);\n")


def patch_tree(root: Path, repo_root: Path) -> None:
    include_dir = root / "cpp/include/tensorrt_llm/batch_manager"
    source_dir = root / "cpp/tensorrt_llm/batch_manager"
    cmake = source_dir / "CMakeLists.txt"
    transfer = source_dir / "kvCacheTransferManager.cpp"
    tracker_header = include_dir / "datasystemRequestTracker.h"
    for path in (cmake, transfer, tracker_header):
        if not path.is_file():
            raise RuntimeError(f"missing prerequisite TensorRT-LLM file: {path}")
    if "currentDataSystemRequestId" not in tracker_header.read_text():
        raise RuntimeError("apply the DataSystem request attribution patch before the KVC burst proxy patch")

    shared, proxy_header, proxy_source = managed_sources(repo_root)
    write_managed(include_dir / "kvcBurstShared.h", shared)
    write_managed(include_dir / "kvcOperationProxy.h", proxy_header)
    write_managed(source_dir / "kvcOperationProxy.cpp", proxy_source)
    replace_once(cmake, "    datasystemRequestTracker.cpp\n",
        "    datasystemRequestTracker.cpp\n    kvcOperationProxy.cpp\n")
    replace_once(transfer,
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n',
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n'
        '#include "tensorrt_llm/batch_manager/kvcOperationProxy.h"\n')

    patch_wrapped_get(transfer, "attributionGet", "kvcBurstGet", "kGet", "1U", (
        "datasystem::Status getRet = kvClient1->Get("
        "std::to_string(BlockKeyHasher::hash(src->getBlockKey())), buffer, 0);",
        "datasystem::Status getRet = kvClient1->Get(key, buffer, 0);",
    ))
    patch_wrapped_get(transfer, "attributionMGet", "kvcBurstMGet", "kMGet",
        "static_cast<uint32_t>(keys.size())", (
            "datasystem::Status getRet = kvClient->Get(keys, buffers, 0);",
            "auto const getRet = kvClient->Get(keys, buffers, 0);",
        ))
    patch_parallel_get(transfer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trtllm_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.trtllm_dir.resolve()
    patch_tree(root, args.repo_root.resolve())
    print(f"TRTLLM_KVC_BURST_PROXY_PATCH_OK root={root}")


if __name__ == "__main__":
    main()
