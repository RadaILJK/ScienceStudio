"""Re-run the documented frozen-BrainIAC baseline and save auditable outputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation import evaluate_features, validate_features
from plot_results import make_plots

BUNDLE = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root, value):
    value = Path(value)
    return value if value.is_absolute() else root / value


def environment():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "monai",
                "nibabel", "matplotlib", "einops", "tqdm", "huggingface_hub", "safetensors"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def find_metadata(root, config):
    for value in config["data"]["participants_candidates"]:
        candidate = resolve_path(root, value)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("participants.tsv not found. Set --project-root or update config.json.")


def prepare_manifest(root, config, out):
    metadata = find_metadata(root, config)
    df = pd.read_csv(metadata, sep="\t", dtype={"participant_id": str})
    if not {"participant_id", "group"} <= set(df.columns):
        raise ValueError("participants.tsv needs participant_id and group columns.")
    df["group"] = df["group"].str.strip().str.upper()
    df = df[df["group"].isin(["HC", "ULD"])].copy()
    if df["participant_id"].isna().any() or df["participant_id"].duplicated().any():
        raise ValueError("Missing or duplicate participant IDs in participants.tsv.")
    data_dir = resolve_path(root, config["data"]["data_dir"])
    rows = []
    for _, row in df.iterrows():  # Preserve original TSV order.
        participant = row["participant_id"]
        files = sorted((data_dir / participant / "anat").glob("*_T1w.nii*"))
        rows.append({"participant_id": participant, "group": row["group"],
            "label": int(row["group"] == "ULD"), "selected_path": str(files[0]) if files else "",
            "n_matching_scans": len(files), "file_present": bool(files)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    write_json(out / "data_provenance.json", {"participants_tsv": str(metadata),
        "participants_sha256": sha256(metadata), "dataset": config["dataset"],
        "selection": "TSV order; first lexicographically sorted T1w in each subject/anat"})
    return manifest


def check_counts(y, config, allow_subset):
    actual = {"HC": int((y == 0).sum()), "ULD": int((y == 1).sum())}
    if not allow_subset and actual != config["data"]["expected_counts"]:
        raise ValueError(f"Expected {config['data']['expected_counts']}, got {actual}. "
                         "See exclusions.csv. Use --allow-subset only for an intentionally different cohort.")


def load_encoder(config, root, checkpoint_override, out):
    import torch
    from monai.networks.nets import ViT
    from safetensors.torch import load_file
    torch.manual_seed(config["seed"])
    device_name = config["model"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = checkpoint_override or config["model"]["checkpoint_path"]
    if checkpoint:
        checkpoint = resolve_path(root, checkpoint)
    else:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(
            repo_id=config["model"]["repo_id"], filename=config["model"]["filename"],
            revision=config["model"]["revision"]))
    model = ViT(**config["model"]["architecture"])
    state = load_file(str(checkpoint), device="cpu")
    if any(key.startswith("backbone.") for key in state):
        original_count = len(state)
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
        if len(state) != original_count:
            raise ValueError("Checkpoint keys collide after removing backbone. prefix.")
    result = model.load_state_dict(state, strict=False)
    allowed = {f"blocks.{i}.norm_cross_attn.{name}" for i in range(config["model"]["architecture"]["num_layers"]) for name in ("weight", "bias")}
    missing, unexpected = set(result.missing_keys), set(result.unexpected_keys)
    if not missing.issubset(allowed) or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing-allowed}; unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    write_json(out / "checkpoint_provenance.json", {
        "repo_id": config["model"]["repo_id"], "requested_revision": config["model"]["revision"],
        "path": str(checkpoint), "sha256": sha256(checkpoint),
        "snapshot_revision_if_available": checkpoint.parent.name if checkpoint.parent.parent.name == "snapshots" else None,
        "allowed_missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
        "n_parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "readout": "output[:, 0, :] = first patch token, not CLS"})
    logging.info("Encoder ready on %s. Allowed missing parameters: %d", device, len(missing))
    return model, device


def extract_features(manifest, config, root, checkpoint_override, out, allow_subset):
    import torch
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Resized, NormalizeIntensityd, ToTensord)
    model, device = load_encoder(config, root, checkpoint_override, out)
    preprocess = Compose([
        LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=config["preprocessing"]["orientation"]),
        Resized(keys=["image"], spatial_size=tuple(config["preprocessing"]["spatial_size"]), mode=config["preprocessing"]["resize_mode"]),
        NormalizeIntensityd(keys=["image"], nonzero=config["preprocessing"]["nonzero"], channel_wise=config["preprocessing"]["channel_wise"]),
        ToTensord(keys=["image"])])
    features, labels, ids, exclusions, selected = [], [], [], [], []
    started = time.perf_counter()
    for _, row in manifest.iterrows():
        try:
            if not row["file_present"]:
                raise FileNotFoundError("T1w scan not found")
            x = preprocess({"image": row["selected_path"]})["image"]
            x = x.unsqueeze(0).float().to(device)
            with torch.inference_mode():
                output, _ = model(x)
                vector = output[:, 0, :].cpu().numpy().ravel()
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError("Expected 768 finite BrainIAC features")
            features.append(vector)
            labels.append(int(row["label"]))
            ids.append(row["participant_id"])
            selected.append({"participant_id": row["participant_id"],
                "path": row["selected_path"], "sha256": sha256(row["selected_path"])})
            logging.info("Features: %s (%d/%d)", row["participant_id"], len(ids), len(manifest))
        except Exception as error:
            exclusions.append({"participant_id": row["participant_id"], "reason": str(error)})
            logging.warning("Excluded %s: %s", row["participant_id"], error)
    pd.DataFrame(exclusions, columns=["participant_id", "reason"]).to_csv(out / "exclusions.csv", index=False)
    write_json(out / "input_image_hashes.json", selected)
    if exclusions and not allow_subset:
        raise ValueError("Some images failed. See exclusions.csv; no classification was run.")
    X, y, ids = validate_features(np.asarray(features), np.asarray(labels), np.asarray(ids), config["cv"]["n_splits"])
    return X, y, ids, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config.json")
    parser.add_argument("--project-root", type=Path, default=BUNDLE.parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embeddings", type=Path, help="Existing NPZ with X, y, ids (same order).")
    parser.add_argument("--checkpoint", help="Local backbone.safetensors; avoids downloading weights.")
    parser.add_argument("--check-inputs", action="store_true", help="Check metadata/files without loading the encoder.")
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly allow a cohort other than 37 HC + 29 ULD.")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.project_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = (args.output_dir or BUNDLE.parent / "results" / "runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler()])
    started = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    write_json(out / "config_resolved.json", {**config, "project_root": str(root),
        "output_dir": str(out), "source_code_sha256": {p.name: sha256(p) for p in [Path(__file__), BUNDLE / "evaluation.py", BUNDLE / "plot_results.py"]}})
    write_json(out / "environment.json", environment())
    try:
        manifest = None
        if args.embeddings:
            with np.load(args.embeddings, allow_pickle=False) as cache:
                X, y, ids = validate_features(cache["X"], cache["y"], cache["ids"], config["cv"]["n_splits"])
            write_json(out / "feature_provenance.json", {"mode": "provided_embeddings",
                "path": str(args.embeddings.resolve()), "sha256": sha256(args.embeddings)})
            try:
                metadata = find_metadata(root, config)
            except FileNotFoundError:
                metadata = None
            if metadata:
                table = pd.read_csv(metadata, sep="\t", dtype={"participant_id": str})
                table["group"] = table["group"].str.strip().str.upper()
                group_map = table.set_index("participant_id")["group"].to_dict()
                if any(group_map.get(i) != ("ULD" if label else "HC") for i, label in zip(ids, y)):
                    raise ValueError("Cache labels/IDs do not match participants.tsv.")
            pd.DataFrame({"participant_id": ids, "label": y}).to_csv(out / "manifest.csv", index=False)
            extraction_seconds = None
        else:
            manifest = prepare_manifest(root, config, out)
            missing = manifest.loc[~manifest["file_present"], ["participant_id"]].copy()
            missing["reason"] = "T1w scan not found"
            missing.to_csv(out / "exclusions.csv", index=False)
            if args.check_inputs:
                write_json(out / "input_check.json", {"n_selected": len(manifest),
                    "n_with_scan": int(manifest["file_present"].sum()), "n_missing": len(missing)})
                if len(missing):
                    raise FileNotFoundError("Missing MRI files. Check manifest.csv and --project-root.")
                logging.info("Input check passed: %d participants", len(manifest))
                write_json(out / "run_status.json", {"status": "inputs_checked_only"})
                return 0
            if len(missing) and not args.allow_subset:
                raise FileNotFoundError("Missing MRI files. No model downloaded; see manifest.csv.")
            X, y, ids, extraction_seconds = extract_features(
                manifest, config, root, args.checkpoint, out, args.allow_subset)
        check_counts(y, config, args.allow_subset)
        if X.shape[1] != 768:
            raise ValueError("This baseline requires exactly 768 BrainIAC features.")
        np.savez_compressed(out / "embeddings.npz", X=X, y=y, ids=ids)
        metrics, probabilities, folds, splits, per_fold = evaluate_features(
            X, y, ids, config["cv"], config["classifier"], config["threshold"])
        metrics["feature_extraction_seconds"] = extraction_seconds
        pd.DataFrame({"participant_id": ids, "label": y, "fold": folds,
            "probability_ULD": probabilities, "prediction": (probabilities >= config["threshold"]).astype(int)}
            ).to_csv(out / "oof_predictions.csv", index=False)
        write_json(out / "splits.json", splits)
        pd.DataFrame({"participant_id": ids, "label": y, "fold": folds}).to_csv(out / "folds.csv", index=False)
        write_json(out / "per_fold_metrics.json", per_fold)
        make_plots(out, y, probabilities, metrics["confusion_matrix"])
        metrics["elapsed_seconds_before_final_write"] = time.perf_counter() - started
        write_json(out / "metrics.json", metrics)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        if freeze.returncode == 0:
            (out / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        write_json(out / "run_status.json", {"status": "completed", "kind": "new_computed_run"})
        logging.info("Accuracy %.3f; BA %.3f; ROC-AUC %.3f", metrics["accuracy"],
                     metrics["balanced_accuracy"], metrics["roc_auc"])
        logging.info("Results saved to %s", out)
        return 0
    except Exception as error:
        write_json(out / "run_status.json", {"status": "failed", "error": str(error)})
        logging.exception("Run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

