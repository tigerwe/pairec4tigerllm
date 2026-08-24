#!/usr/bin/env python3
"""Patch TensorRT-LLM DataSystem Get paths with the PaiRec KVC burst proxy."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "PAIREC_KVC_BURST_PROXY_V6"
LEGACY_MARKERS = (
    "PAIREC_KVC_BURST_PROXY_V5", "PAIREC_KVC_BURST_PROXY_V4",
    "PAIREC_KVC_BURST_PROXY_V3", "PAIREC_KVC_BURST_PROXY_V2")


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
    api: str, key_count: str, pressure_client: str | None, calls: tuple[str, ...]) -> None:
    text = transfer.read_text()
    attribution = f"auto const {attribution_prefix}Token = beginDataSystemOperation(DataSystemOperation::kGet);"
    if text.count(attribution) != 1:
        raise RuntimeError(f"missing exact attribution anchor for {proxy_prefix} in {transfer}")
    if f"auto const {proxy_prefix}Token =" not in text:
        begin = (
            f"auto const {proxy_prefix}RequestId = currentDataSystemRequestId();\n"
            f"            auto const {proxy_prefix}Token = pairec::kvc_burst::beginBusinessGet(\n"
            f"                {proxy_prefix}RequestId.value_or(\"\"), "
            f"pairec::kvc_burst::BusinessApi::{api}, {key_count});\n")
        replace_once(transfer, attribution, begin + f"            {attribution}")

    text = transfer.read_text()
    if pressure_client is not None and f"auto {proxy_prefix}Pressure =" not in text:
        capture = pressure_client
        pressure = (
            f"            auto {proxy_prefix}Pressure = "
            "pairec::kvc_burst::beginInProcessPressure(\n"
            f"                {proxy_prefix}Token, [{capture}](uint32_t, std::string const& pressureKey) {{\n"
            "                    datasystem::Optional<datasystem::Buffer> pressureBuffer;\n"
            f"                    auto const pressureStatus = {pressure_client}->Get(pressureKey, pressureBuffer, 0);\n"
            "                    return !pressureStatus.IsError() && static_cast<bool>(pressureBuffer);\n"
            "                });\n")
        replace_once(transfer, attribution, pressure + f"            {attribution}")

    text = transfer.read_text()
    if f"auto const {proxy_prefix}EndedNs =" not in text:
        matches = [call for call in calls if text.count(call) == 1]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one DataSystem Get call for {proxy_prefix} in {transfer}, found {len(matches)}")
        call = matches[0]
        replace_once(transfer, call,
            f"{call}\n"
            f"            auto const {proxy_prefix}EndedNs = pairec::kvc_burst::MonotonicNs();")

    attribution_finish = (
        f"finishDataSystemOperation({attribution_prefix}Token, DataSystemOperation::kGet, "
        f"{attribution_prefix}Us, getRet.IsError());")
    business_finish = (f"pairec::kvc_burst::finishBusinessGet(\n"
                       f"                {proxy_prefix}Token, !getRet.IsError(), {proxy_prefix}EndedNs);")
    text = transfer.read_text()
    if business_finish not in text:
        finish = attribution_finish
        if pressure_client is not None:
            finish += f"\n            {proxy_prefix}Pressure.finish();"
        finish += f"\n            {business_finish}"
        replace_once(transfer, attribution_finish, finish)
    elif pressure_client is not None and f"{proxy_prefix}Pressure.finish();" not in text:
        replace_once(transfer, business_finish,
            f"{proxy_prefix}Pressure.finish();\n            {business_finish}")


def patch_parallel_get(transfer: Path) -> None:
    text = transfer.read_text()
    if "TRTLLM_DATASYSTEM_PARALLEL_GET" not in text:
        return
    loop = "                for (auto const& key : keys)\n"
    if "kvcBurstParallelGetToken" not in text:
        begin = (
            "                auto const kvcBurstParallelGetRequestId = currentDataSystemRequestId();\n"
            "                auto const kvcBurstParallelGetToken = pairec::kvc_burst::beginBusinessGet(\n"
            "                    kvcBurstParallelGetRequestId.value_or(\"\"),\n"
            "                    pairec::kvc_burst::BusinessApi::kParallelGet, keys.size());\n"
            + loop)
        replace_once(transfer, loop, begin)
    text = transfer.read_text()
    if "kvcBurstParallelGetPressure" not in text:
        pressure = (
            "                auto kvcBurstParallelGetPressure = "
            "pairec::kvc_burst::beginInProcessPressure(\n"
            "                    kvcBurstParallelGetToken, "
            "[kvClient](uint32_t, std::string const& pressureKey) {\n"
            "                        datasystem::Optional<datasystem::Buffer> pressureBuffer;\n"
            "                        auto const pressureStatus = kvClient->Get(pressureKey, pressureBuffer, 0);\n"
            "                        return !pressureStatus.IsError() && static_cast<bool>(pressureBuffer);\n"
            "                    });\n")
        replace_once(transfer, loop, pressure + loop)
    completion = (
        "                activeParallelGetCallsAfter = "
        "gActiveParallelGetCalls.fetch_sub(1, std::memory_order_relaxed) - 1;\n")
    business_finish = (
        "pairec::kvc_burst::finishBusinessGet(kvcBurstParallelGetToken, !getFailed);")
    text = transfer.read_text()
    if business_finish not in text:
        replace_once(transfer, completion,
            completion
            + "                kvcBurstParallelGetPressure.finish();\n"
            + f"                {business_finish}\n")
    elif "kvcBurstParallelGetPressure.finish();" not in text:
        replace_once(transfer, business_finish,
            "kvcBurstParallelGetPressure.finish();\n                " + business_finish)


def patch_add_token_observation(manager: Path) -> None:
    replace_once(manager,
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n',
        '#include "tensorrt_llm/batch_manager/datasystemRequestTracker.h"\n'
        '#include "tensorrt_llm/batch_manager/kvcOperationProxy.h"\n')
    start_anchor = (
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    auto const attributionSequenceLookupStarted = attributionEnabled\n")
    if "kvcBurstAddTokenRequestId" not in manager.read_text():
        replace_once(manager, start_anchor,
            "    DataSystemRequestScope attributionScope(clientId);\n"
            "    auto const kvcBurstAddTokenRequestId = currentDataSystemRequestId();\n"
            "    pairec::kvc_burst::observeAddTokenStart(\n"
            "        kvcBurstAddTokenRequestId.value_or(\"\"));\n"
            "    auto const attributionSequenceLookupStarted = attributionEnabled\n")
    end_anchor = (
        "        recordDataSystemRequestPhase(clientId, DataSystemRequestPhase::kAddToken, attributionAddTokenUs,\n"
        "            attributionLookupUs, attributionSequenceLookupUs, attributionKvUpdateUs);\n")
    if "observeAddTokenEnd" not in manager.read_text():
        replace_once(manager, end_anchor,
            "        pairec::kvc_burst::observeAddTokenEnd(\n"
            "            kvcBurstAddTokenRequestId.value_or(\"\"));\n"
            + end_anchor)

    remove_start_anchor = (
        "    DataSystemRequestScope attributionScope(clientId);\n"
        "    TLLM_LOG_TRACE(\"[%s]::%s start\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n")
    if "kvcBurstRemoveSequenceRequestId" not in manager.read_text():
        replace_once(manager, remove_start_anchor,
            "    DataSystemRequestScope attributionScope(clientId);\n"
            "    auto const kvcBurstRemoveSequenceRequestId = currentDataSystemRequestId();\n"
            "    TLLM_LOG_TRACE(\"[%s]::%s start\", isCrossKv() ? \"CROSS\" : \"SELF\", __PRETTY_FUNCTION__);\n")
    remove_end_anchor = "    if (clientId) finishDataSystemRequest(*clientId);\n"
    if "observeBusinessRequestComplete" not in manager.read_text():
        replace_once(manager, remove_end_anchor,
            "    pairec::kvc_burst::observeBusinessRequestComplete(\n"
            "        kvcBurstRemoveSequenceRequestId.value_or(\"\"));\n"
            + remove_end_anchor)


def patch_tree(root: Path, repo_root: Path) -> None:
    include_dir = root / "cpp/include/tensorrt_llm/batch_manager"
    source_dir = root / "cpp/tensorrt_llm/batch_manager"
    cmake = source_dir / "CMakeLists.txt"
    transfer = source_dir / "kvCacheTransferManager.cpp"
    manager = source_dir / "kvCacheManager.cpp"
    tracker_header = include_dir / "datasystemRequestTracker.h"
    for path in (cmake, transfer, manager, tracker_header):
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

    # Persistent workers retain this callback after the Get stack returns.
    text = transfer.read_text()
    text = text.replace("[&kvClient1](uint32_t, std::string const& pressureKey)",
                        "[kvClient1](uint32_t, std::string const& pressureKey)")
    text = text.replace("[&kvClient](uint32_t, std::string const& pressureKey)",
                        "[kvClient](uint32_t, std::string const& pressureKey)")
    transfer.write_text(text)

    patch_wrapped_get(transfer, "attributionGet", "kvcBurstGet", "kGet", "1U", "kvClient1", (
        "datasystem::Status getRet = kvClient1->Get("
        "std::to_string(BlockKeyHasher::hash(src->getBlockKey())), buffer, 0);",
        "datasystem::Status getRet = kvClient1->Get(key, buffer, 0);",
    ))
    patch_wrapped_get(transfer, "attributionMGet", "kvcBurstMGet", "kMGet",
        "static_cast<uint32_t>(keys.size())", "kvClient", (
            "datasystem::Status getRet = kvClient->Get(keys, buffers, 0);",
            "auto const getRet = kvClient->Get(keys, buffers, 0);",
        ))
    patch_parallel_get(transfer)
    patch_add_token_observation(manager)


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
