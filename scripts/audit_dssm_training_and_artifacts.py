#!/usr/bin/env python3
"""Audit DSSM training scope, artifact integrity, and vocabulary coverage."""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dssm.model import DSSM


HIST_COLUMNS = [f"hist_{index}" for index in range(1, 11)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dssm-dir", required=True)
    parser.add_argument("--deepfm-dir", default="")
    parser.add_argument("--csv-path", default="")
    parser.add_argument(
        "--csv-scan-rows",
        default="5000000",
        help="Rows used for sampled OOV checks; use 0 to skip or all for a full scan",
    )
    parser.add_argument("--pipeline-script", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ratio(numerator, denominator):
    if not denominator:
        return None
    return numerator / denominator


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_pipeline_settings(path):
    if not path or not os.path.isfile(path):
        return {}
    text = Path(path).read_text(encoding="utf-8")
    options = {
        "train_rows": "train_rows",
        "vocab_rows": "vocab_rows",
        "profile_rows": "profile_rows",
        "batch_size": "batch_size",
        "epochs": "epochs",
    }
    result = {}
    for option, key in options.items():
        match = re.search(rf"--{option}\s+([0-9]+)", text)
        if match:
            result[key] = int(match.group(1))
    return result


def inspect_checkpoint(checkpoint, vocab):
    checks = []
    config = checkpoint.get("config", {})
    expected_sizes = {
        "user": len(vocab.get("user2idx", {})) + 1,
        "item": len(vocab.get("item2idx", {})) + 1,
        "cat": len(vocab.get("cat2idx", {})) + 1,
        "gender": len(vocab.get("gender2idx", {})) + 1,
        "age": len(vocab.get("age2idx", {})) + 1,
    }
    checks.append({
        "name": "checkpoint_vocab_sizes_match",
        "pass": config.get("vocab_sizes") == expected_sizes,
        "expected": expected_sizes,
        "actual": config.get("vocab_sizes"),
    })
    state = checkpoint.get("model")
    state_valid = isinstance(state, dict) and bool(state)
    if state_valid:
        state_valid = all(
            torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        )
    checks.append({"name": "checkpoint_tensors_finite", "pass": state_valid})
    checks.append({
        "name": "checkpoint_loss_finite",
        "pass": finite_number(checkpoint.get("avg_loss")),
        "value": checkpoint.get("avg_loss"),
    })

    load_error = None
    smoke_finite = False
    try:
        model = DSSM(
            expected_sizes,
            embed_dim=int(config["embed_dim"]),
            out_dim=int(config["out_dim"]),
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            user = model.user_tower(
                torch.tensor([min(1, expected_sizes["user"] - 1)]),
                torch.tensor([min(1, expected_sizes["gender"] - 1)]),
                torch.tensor([min(1, expected_sizes["age"] - 1)]),
                torch.zeros((1, 10), dtype=torch.long),
            )
            item = model.item_tower(
                torch.tensor([min(1, expected_sizes["item"] - 1)]),
                torch.tensor([min(1, expected_sizes["cat"] - 1)]),
            )
        smoke_finite = bool(torch.isfinite(user).all() and torch.isfinite(item).all())
    except Exception as exc:  # Report the precise artifact incompatibility.
        load_error = repr(exc)
    checks.append({
        "name": "checkpoint_model_load_and_forward",
        "pass": smoke_finite,
        "error": load_error,
    })
    return checks, expected_sizes


def inspect_vectors(export_dir, checkpoint, vocab):
    vectors_path = export_dir / "item_vectors.npy"
    ids_path = export_dir / "item_ids.json"
    vectors = np.load(vectors_path, mmap_mode="r")
    item_ids = load_json(ids_path)
    item_vocab_ids = {int(value) for value in vocab["item2idx"]}
    exported_ids = {int(value) for value in item_ids}
    out_dim = int(checkpoint.get("config", {}).get("out_dim", -1))

    finite = bool(np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([])
    norm_max_error = float(np.max(np.abs(norms - 1.0))) if len(norms) else None
    checks = [
        {
            "name": "vector_shape_matches_ids",
            "pass": vectors.ndim == 2 and vectors.shape[0] == len(item_ids),
            "shape": list(vectors.shape),
            "item_id_count": len(item_ids),
        },
        {
            "name": "vector_dimension_matches_checkpoint",
            "pass": vectors.ndim == 2 and vectors.shape[1] == out_dim,
            "expected": out_dim,
            "actual": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        },
        {"name": "vectors_finite", "pass": finite},
        {
            "name": "vectors_l2_normalized",
            "pass": norm_max_error is not None and norm_max_error <= 1e-4,
            "max_abs_error": norm_max_error,
        },
        {
            "name": "exported_item_ids_unique",
            "pass": len(exported_ids) == len(item_ids),
        },
        {
            "name": "exported_items_match_vocab",
            "pass": exported_ids == item_vocab_ids,
            "vocab_items": len(item_vocab_ids),
            "exported_items": len(exported_ids),
        },
    ]

    rng = np.random.default_rng(20260807)
    sample_count = min(len(vectors), 10000)
    collapse = {"sample_count": sample_count}
    if sample_count >= 2 and finite:
        sample_indices = rng.choice(len(vectors), sample_count, replace=False)
        sample = np.asarray(vectors[sample_indices], dtype=np.float32)
        pair_count = min(10000, sample_count * 2)
        left = rng.integers(0, sample_count, size=pair_count)
        right = rng.integers(0, sample_count, size=pair_count)
        same = left == right
        right[same] = (right[same] + 1) % sample_count
        cosine = np.sum(sample[left] * sample[right], axis=1)
        dim_std = np.std(sample, axis=0)
        collapse.update({
            "pair_cosine_mean": float(np.mean(cosine)),
            "pair_cosine_p95": float(np.percentile(cosine, 95)),
            "pair_cosine_p99": float(np.percentile(cosine, 99)),
            "per_dimension_std_mean": float(np.mean(dim_std)),
            "per_dimension_std_min": float(np.min(dim_std)),
            "suspected_collapse": bool(
                np.percentile(cosine, 99) > 0.999 or np.mean(dim_std) < 1e-5
            ),
        })
    else:
        collapse["suspected_collapse"] = True
    return checks, collapse, exported_ids


def full_data_evidence(deepfm_dir):
    evidence = {
        "available": False,
        "row_count": None,
        "target_item_ids": None,
        "deepfm_best_val_auc": None,
        "deepfm_best_val_logloss": None,
        "feature_vocab_path": None,
    }
    if not deepfm_dir or not deepfm_dir.is_dir():
        return evidence
    summary_path = deepfm_dir / "training_summary.json"
    categories_path = deepfm_dir / "item_categories.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        history = summary.get("history", [])
        totals = [
            int(row.get("train_examples", 0)) + int(row.get("val_examples", 0))
            for row in history
        ]
        evidence["row_count"] = max(totals) if totals else None
        evidence["deepfm_best_val_auc"] = summary.get("best_val_auc")
        evidence["deepfm_best_val_logloss"] = summary.get("best_val_logloss")
        evidence["available"] = True
    if categories_path.is_file():
        evidence["target_item_ids"] = {int(value) for value in load_json(categories_path)}
        evidence["available"] = True
    vocab_path = deepfm_dir / "feature_vocab.json"
    if vocab_path.is_file():
        evidence["feature_vocab_path"] = str(vocab_path)
    return evidence


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def update_oov(counter, values, known):
    valid = values >= 0
    valid_count = int(valid.sum())
    counter["valid"] += valid_count
    if valid_count:
        counter["oov"] += int((~np.isin(values[valid], known)).sum())


def scan_csv_oov(csv_path, scan_rows, vocab):
    if scan_rows == "0" or not csv_path or not os.path.isfile(csv_path):
        return {"performed": False, "requested_rows": scan_rows}
    limit = None if scan_rows.lower() == "all" else int(scan_rows)
    if limit is not None and limit < 0:
        raise ValueError("csv-scan-rows must be 0, all, or a positive integer")
    known = {
        "user": np.fromiter((int(key) for key in vocab["user2idx"]), dtype=np.int64),
        "item": np.fromiter((int(key) for key in vocab["item2idx"]), dtype=np.int64),
        "cat": np.fromiter((int(key) for key in vocab["cat2idx"]), dtype=np.int64),
        "gender": np.fromiter((int(key) for key in vocab["gender2idx"]), dtype=np.int64),
        "age": np.fromiter((int(key) for key in vocab["age2idx"]), dtype=np.int64),
    }
    counters = {
        key: {"valid": 0, "oov": 0}
        for key in ["user", "target_item", "history_item", "category", "gender", "age"]
    }
    usecols = ["user_id", "item_id", "video_category", "gender", "age"] + HIST_COLUMNS
    scanned = 0
    with pd.read_csv(csv_path, usecols=usecols, chunksize=250000,
                     na_values=["\\N"]) as chunks:
        for chunk in chunks:
            if limit is not None and scanned + len(chunk) > limit:
                chunk = chunk.iloc[:limit - scanned]
            update_oov(counters["user"], numeric_values(chunk["user_id"]), known["user"])
            update_oov(counters["target_item"], numeric_values(chunk["item_id"]), known["item"])
            update_oov(counters["category"], numeric_values(chunk["video_category"]), known["cat"])
            update_oov(counters["gender"], numeric_values(chunk["gender"]), known["gender"])
            update_oov(counters["age"], numeric_values(chunk["age"]), known["age"])
            history = np.concatenate([numeric_values(chunk[column]) for column in HIST_COLUMNS])
            update_oov(counters["history_item"], history, known["item"])
            scanned += len(chunk)
            if limit is not None and scanned >= limit:
                break
    for counter in counters.values():
        counter["oov_rate"] = ratio(counter["oov"], counter["valid"])
    return {
        "performed": True,
        "requested_rows": scan_rows,
        "scanned_rows": scanned,
        "features": counters,
    }


def audit(args):
    dssm_dir = Path(args.dssm_dir)
    export_dir = dssm_dir / "export"
    required = [
        dssm_dir / "vocab.json",
        dssm_dir / "dssm_model.pt",
        export_dir / "item_vectors.npy",
        export_dir / "item_ids.json",
        export_dir / "item_categories.json",
        export_dir / "user_profiles.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "result": "FAIL",
            "classification": "DSSM_ARTIFACTS_MISSING",
            "missing": missing,
        }

    vocab = load_json(dssm_dir / "vocab.json")
    checkpoint = load_checkpoint(dssm_dir / "dssm_model.pt")
    checkpoint_checks, vocab_sizes = inspect_checkpoint(checkpoint, vocab)
    vector_checks, collapse, exported_ids = inspect_vectors(export_dir, checkpoint, vocab)
    checks = checkpoint_checks + vector_checks

    pipeline = parse_pipeline_settings(args.pipeline_script)
    deepfm = full_data_evidence(Path(args.deepfm_dir) if args.deepfm_dir else None)
    full_items = deepfm.pop("target_item_ids")
    deepfm_vocab_matches = None
    if deepfm.get("feature_vocab_path"):
        deepfm_vocab_matches = load_json(deepfm["feature_vocab_path"]) == vocab
    coverage = {
        "dssm_vocab_sizes": vocab_sizes,
        "exported_item_count": len(exported_ids),
        "user_profile_count": len(load_json(export_dir / "user_profiles.json")),
        "dssm_item_category_count": len(load_json(export_dir / "item_categories.json")),
        "full_target_item_count": len(full_items) if full_items is not None else None,
        "full_target_item_covered": len(exported_ids & full_items) if full_items is not None else None,
        "full_target_item_coverage": ratio(
            len(exported_ids & full_items), len(full_items)
        ) if full_items is not None else None,
        "deepfm_feature_vocab_matches_dssm": deepfm_vocab_matches,
    }
    train_rows = pipeline.get("train_rows")
    full_rows = deepfm.get("row_count")
    scope = {
        "pipeline": pipeline,
        "full_data_rows_from_deepfm": full_rows,
        "train_to_full_ratio": ratio(train_rows, full_rows),
        "expected_batches_per_epoch": (
            train_rows // pipeline["batch_size"]
            if train_rows and pipeline.get("batch_size") else None
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_avg_loss": checkpoint.get("avg_loss"),
        "has_heldout_quality_metrics": False,
    }
    csv_oov = scan_csv_oov(args.csv_path, args.csv_scan_rows, vocab)

    failures = [check["name"] for check in checks if not check["pass"]]
    warnings = []
    if scope["train_to_full_ratio"] is not None and scope["train_to_full_ratio"] < 0.95:
        warnings.append("DSSM_TRAINING_SCOPE_PARTIAL")
    if not scope["has_heldout_quality_metrics"]:
        warnings.append("DSSM_NO_HELDOUT_QUALITY_METRICS")
    item_coverage = coverage["full_target_item_coverage"]
    if item_coverage is not None and item_coverage < 0.95:
        warnings.append("DSSM_TARGET_ITEM_COVERAGE_LOW")
    if collapse.get("suspected_collapse"):
        warnings.append("DSSM_VECTOR_COLLAPSE_SUSPECTED")
    if csv_oov.get("performed"):
        for name, metrics in csv_oov["features"].items():
            if (metrics["oov_rate"] or 0.0) > 0.05:
                warnings.append(f"DSSM_SAMPLED_{name.upper()}_OOV_HIGH")

    if failures:
        result = "FAIL"
        classification = "DSSM_ARTIFACT_INTEGRITY_FAILED"
    elif warnings:
        result = "WARN"
        classification = warnings[0]
    else:
        result = "PASS"
        classification = "DSSM_ARTIFACT_AND_SCOPE_OK"
    return {
        "result": result,
        "classification": classification,
        "failures": failures,
        "warnings": warnings,
        "artifact_checks": checks,
        "training_scope": scope,
        "coverage": coverage,
        "vector_distribution": collapse,
        "csv_oov_sample": csv_oov,
        "deepfm_full_data_evidence": deepfm,
    }


def percentage(value):
    return "n/a" if value is None else f"{value * 100:.3f}%"


def print_report(report, output):
    print("\n== DSSM artifact integrity ==")
    for check in report.get("artifact_checks", []):
        print(f"{check['name']}={'PASS' if check['pass'] else 'FAIL'}")
    scope = report.get("training_scope", {})
    print("\n== Training scope ==")
    for key, value in scope.get("pipeline", {}).items():
        print(f"configured_{key}={value}")
    print(f"full_data_rows={scope.get('full_data_rows_from_deepfm')}")
    print(f"train_to_full_ratio={percentage(scope.get('train_to_full_ratio'))}")
    print(f"expected_batches_per_epoch={scope.get('expected_batches_per_epoch')}")
    print(f"checkpoint_epoch={scope.get('checkpoint_epoch')}")
    print(f"checkpoint_avg_loss={scope.get('checkpoint_avg_loss')}")
    print("heldout_quality_metrics=ABSENT")

    coverage = report.get("coverage", {})
    print("\n== Coverage ==")
    print(f"exported_item_count={coverage.get('exported_item_count')}")
    print(f"full_target_item_count={coverage.get('full_target_item_count')}")
    print(f"full_target_item_coverage={percentage(coverage.get('full_target_item_coverage'))}")
    print(f"user_profile_count={coverage.get('user_profile_count')}")
    print(f"dssm_item_category_count={coverage.get('dssm_item_category_count')}")
    print(f"deepfm_feature_vocab_matches_dssm={coverage.get('deepfm_feature_vocab_matches_dssm')}")

    collapse = report.get("vector_distribution", {})
    print("\n== Vector distribution ==")
    for key in ["sample_count", "pair_cosine_mean", "pair_cosine_p95",
                "pair_cosine_p99", "per_dimension_std_mean", "suspected_collapse"]:
        print(f"{key}={collapse.get(key)}")

    sample = report.get("csv_oov_sample", {})
    print("\n== CSV OOV sample ==")
    print(f"performed={sample.get('performed')} scanned_rows={sample.get('scanned_rows')}")
    for name, values in sample.get("features", {}).items():
        print(f"{name}_oov={values['oov']}/{values['valid']} ({percentage(values['oov_rate'])})")

    print("\n== Diagnosis ==")
    print(f"result={report.get('result')}")
    print(f"classification={report.get('classification')}")
    print("warnings=" + ",".join(report.get("warnings", [])))
    print("failures=" + ",".join(report.get("failures", [])))
    print(f"report={output}")
    print("DSSM_TRAINING_AUDIT_COMPLETE")


def main():
    args = parse_args()
    report = audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_report(report, output)
    return 1 if report.get("result") == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
