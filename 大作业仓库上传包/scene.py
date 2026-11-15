"""Evaluate image corruptions and classify an already cropped traffic sign.

Examples:
    python scene.py evaluate --device cuda
    python scene.py classify results/example_input.png --device cpu

The application model and confidence threshold are selected on validation only.
The 99% threshold criterion is an empirical validation target, not a safety
guarantee, probability calibration guarantee, or evidence of real-road readiness.
The program classifies one crop; it does not detect signs in a complete scene.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import gaussian_blur

from data import load_data
from models import VARIANTS, build_model
from train import configure, evaluate, preprocess, save_json, sha256


ROOT = Path(__file__).resolve().parent
THRESHOLD_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
TARGET_SELECTIVE_ACCURACY = 0.99
CONDITIONS = {
    "clean": None,
    "brightness_0.5": lambda x: (x * 0.5).clamp(0, 1),
    "brightness_1.5": lambda x: (x * 1.5).clamp(0, 1),
    "blur_sigma1": lambda x: gaussian_blur(x, kernel_size=[5, 5], sigma=[1.0, 1.0]),
    "blur_sigma2": lambda x: gaussian_blur(x, kernel_size=[5, 5], sigma=[2.0, 2.0]),
}


def selective_metrics(probabilities, labels, threshold, reject_all=False):
    """Return accepted-sample accuracy; None represents an empty accepted set."""
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = np.zeros(len(labels), dtype=bool) if reject_all else confidence >= threshold
    count = int(accepted.sum())
    correct = int((predictions[accepted] == labels[accepted]).sum())
    return {
        "count": int(len(labels)), "accepted": count, "rejected": int(len(labels) - count),
        "coverage": float(count / len(labels)), "accepted_correct": correct,
        "accepted_errors": count - correct,
        "accuracy_on_accepted": float(correct / count) if count else None,
    }


def select_threshold(probabilities, labels):
    grid = []
    for threshold in THRESHOLD_GRID:
        row = {"threshold": threshold, **selective_metrics(probabilities, labels, threshold)}
        row["meets_target"] = (
            row["accepted"] > 0 and row["accepted_correct"] * 100 >= row["accepted"] * 99
        )
        grid.append(row)
    eligible = [row for row in grid if row["meets_target"]]
    if not eligible:
        return None, True, selective_metrics(probabilities, labels, None, True), grid
    # Highest coverage first; lowest threshold breaks an equal-coverage tie.
    chosen = min(eligible, key=lambda row: (-row["accepted"], row["threshold"]))
    return chosen["threshold"], False, selective_metrics(probabilities, labels, chosen["threshold"]), grid


def _device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu.")
    return device


def _load_model(checkpoint_path, variant, device, expected_sha256=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}; finish train.py first.")
    digest = sha256(checkpoint_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint does not match recorded experiment: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != variant:
        raise ValueError("Checkpoint variant does not match selected variant")
    model = build_model(variant, num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    return model, checkpoint, digest


@torch.inference_mode()
def measure_latency(model, normalized_image, warmup=10, repeats=50):
    """FP32 batch-one forward latency, excluding input preprocessing/transfer."""
    model.eval()
    device = normalized_image.device
    for _ in range(warmup):
        model(normalized_image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(normalized_image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
        "minimum": float(min(times)), "samples": times, "unit": "ms",
        "batch_size": 1, "precision": "FP32", "warmup": warmup, "repeats": repeats,
        "scope": "Model forward only; excludes preprocessing and host-to-device copy",
        "cuda_synchronized": device.type == "cuda",
    }


def evaluate_scene(args):
    configure(42, args.threads)
    device = _device(args.device)
    result_dir, checkpoint_dir = Path(args.result_dir), Path(args.checkpoint_dir)
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Missing results/summary.json; finish train.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = set(VARIANTS) - set(summary["models"])
    if missing:
        raise ValueError(f"Training is incomplete; missing variants: {sorted(missing)}")
    candidates = [
        {"variant": name, "validation_accuracy": summary["models"][name]["validation"]["accuracy"],
         "parameters": summary["models"][name]["profile"]["parameters"]}
        for name in ("depthwise", "depthwise_se")
    ]
    chosen_variant = min(candidates, key=lambda item: (-item["validation_accuracy"], item["parameters"]))["variant"]
    data = load_data(args.data_dir, image_size=64, seed=42, val_fraction=0.2)
    if data["metadata"]["split_digest"] != summary["data"]["split_digest"]:
        raise ValueError("Dataset split differs from the training experiment")
    mean = torch.tensor(data["mean"], device=device)[None, :, None, None]
    std = torch.tensor(data["std"], device=device)[None, :, None, None]
    test_images = torch.as_tensor(np.ascontiguousarray(data["test_images"]), device=device)
    test_labels = torch.as_tensor(data["test_labels"], dtype=torch.long, device=device)
    val_images = torch.as_tensor(np.ascontiguousarray(data["val_images"]), device=device)
    val_labels = torch.as_tensor(data["val_labels"], dtype=torch.long, device=device)
    result = {
        "schema_version": 1, "chosen_variant": chosen_variant,
        "threshold": None, "reject_all": None,
        "selection": {
            "model_rule": "Maximum clean validation accuracy among depthwise and depthwise_se; fewer parameters breaks ties",
            "candidates": candidates,
            "threshold_rule": "Maximum validation coverage with accepted-sample accuracy >= 0.99; lowest threshold breaks ties; otherwise reject all",
            "target_validation_accuracy_on_accepted": TARGET_SELECTIVE_ACCURACY,
            "threshold_grid": THRESHOLD_GRID,
            "uses_test_for_selection": False,
        },
        "environment": {
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "threads": args.threads, "precision": "FP32", "tf32": False,
        },
        "protocol": {
            "dataset": "GTSRB official test split", "count": len(test_labels),
            "input_size": [3, 64, 64], "evaluation_batch_size": args.batch_size,
            "conditions": {
                "clean": "Unmodified resized RGB crop",
                "brightness_0.5": "Multiply RGB [0,1] by 0.5 and clamp [0,1] before normalization",
                "brightness_1.5": "Multiply RGB [0,1] by 1.5 and clamp [0,1] before normalization",
                "blur_sigma1": "Gaussian blur: kernel 5x5, sigma 1.0, reflect padding, before normalization",
                "blur_sigma2": "Gaussian blur: kernel 5x5, sigma 2.0, reflect padding, before normalization",
            },
            "split_digest": data["metadata"]["split_digest"],
            "source_sha256": {name: sha256(ROOT / name) for name in ["models.py", "data.py", "train.py", "scene.py"]},
            "limitations": [
                "Synthetic blur and brightness shifts are not a real-road safety validation.",
                "The threshold is selected by empirical validation accuracy and is not a calibrated risk bound.",
                "Single cropped German traffic-sign classification; no scene detection or automatic vehicle control.",
                "Latency is specific to this software/hardware run, not inferred directly from MACs.",
            ],
        },
        "models": {}, "application": {"validation": None, "test_conditions": {}},
    }
    chosen_clean_probabilities = None
    for variant in VARIANTS:
        training_result = summary["models"][variant]
        model, checkpoint, digest = _load_model(
            checkpoint_dir / f"{variant}.pt", variant, device, training_result["checkpoint_sha256"],
        )
        if checkpoint["split_digest"] != data["metadata"]["split_digest"]:
            raise ValueError("Checkpoint has a different validation split")
        if checkpoint["mean"] != data["mean"] or checkpoint["std"] != data["std"]:
            raise ValueError("Checkpoint normalization differs from this dataset")
        model_result = {"checkpoint_sha256": digest, "best_epoch": checkpoint["epoch"], "conditions": {}}
        if variant == chosen_variant:
            _, probabilities = evaluate(
                model, val_images, val_labels, mean, std, args.batch_size, return_probabilities=True,
            )
            threshold, reject_all, validation_selective, grid = select_threshold(probabilities, data["val_labels"])
            result["threshold"], result["reject_all"] = threshold, reject_all
            result["selection"]["validation_grid_results"] = grid
            result["application"]["validation"] = validation_selective
            result["application"]["checkpoint_sha256"] = digest
        clean_accuracy = None
        for condition, transform in CONDITIONS.items():
            metrics, probabilities = evaluate(
                model, test_images, test_labels, mean, std, args.batch_size,
                return_probabilities=True, transform=transform,
            )
            if condition == "clean":
                clean_accuracy = metrics["accuracy"]
                model_result["clean_correct_matches_training"] = metrics["correct"] == training_result["test"]["correct"]
                model_result["training_record_clean_correct"] = training_result["test"]["correct"]
            compact = {key: metrics[key] for key in
                       ["count", "correct", "accuracy", "macro_f1", "balanced_accuracy", "cross_entropy"]}
            compact["decline_pp"] = (clean_accuracy - metrics["accuracy"]) * 100
            model_result["conditions"][condition] = compact
            if variant == chosen_variant:
                result["application"]["test_conditions"][condition] = selective_metrics(
                    probabilities, data["test_labels"], result["threshold"], result["reject_all"],
                )
                if condition == "clean":
                    chosen_clean_probabilities = probabilities
            print(json.dumps({"event": "scene_condition", "variant": variant,
                              "condition": condition, **compact}), flush=True)
        normalized_image = preprocess(test_images[:1], mean, std)
        model_result["latency_ms"] = measure_latency(model, normalized_image)
        result["models"][variant] = model_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Use the first official test sample, without choosing it based on success.
    index = 0
    example_path = result_dir / "example_input.png"
    Image.fromarray(np.asarray(data["test_images"][index]).transpose(1, 2, 0)).save(example_path)
    prediction = int(chosen_clean_probabilities[index].argmax())
    confidence = float(chosen_clean_probabilities[index].max())
    requires_review = result["reject_all"] or confidence < result["threshold"]
    truth = int(data["test_labels"][index])
    result["example"] = {
        "file": "example_input.png", "test_index": index, "source_path": data["test_paths"][index],
        "true_class_id": truth, "true_class_name": data["classes"][truth], "true_class_name_zh": data["classes_zh"][truth],
        "predicted_class_id": prediction, "predicted_class_name": data["classes"][prediction],
        "predicted_class_name_zh": data["classes_zh"][prediction], "confidence": confidence,
        "requires_review": bool(requires_review), "correct": prediction == truth,
    }
    result["summary_sha256"] = sha256(summary_path)
    save_json(result_dir / "scene.json", result)
    print(json.dumps({"event": "scene_complete", "chosen_variant": chosen_variant,
                      "threshold": result["threshold"], "reject_all": result["reject_all"],
                      "validation": result["application"]["validation"]}), flush=True)


@torch.inference_mode()
def classify_image(args):
    configure(42, args.threads)
    device = _device(args.device)
    result_dir = Path(args.result_dir)
    scene_path = result_dir / "scene.json"
    if not scene_path.is_file():
        raise FileNotFoundError("Missing results/scene.json; run scene.py evaluate first.")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    variant = scene["chosen_variant"]
    model, checkpoint, digest = _load_model(
        Path(args.checkpoint_dir) / f"{variant}.pt", variant, device,
        scene["application"]["checkpoint_sha256"],
    )
    with Image.open(args.image) as image:
        resized = image.convert("RGB").resize((checkpoint["input_size"], checkpoint["input_size"]), Image.Resampling.BILINEAR)
        array = np.array(resized, dtype=np.uint8).transpose(2, 0, 1).copy()
    images = torch.as_tensor(array[None], device=device)
    mean = torch.tensor(checkpoint["mean"], device=device)[None, :, None, None]
    std = torch.tensor(checkpoint["std"], device=device)[None, :, None, None]
    probabilities = model(preprocess(images, mean, std)).softmax(1)[0]
    class_id = int(probabilities.argmax().item())
    confidence = float(probabilities[class_id].item())
    requires_review = scene["reject_all"] or confidence < scene["threshold"]
    prediction = {
        "image": str(Path(args.image)), "variant": variant, "class_id": class_id,
        "class_name": checkpoint["classes"][class_id], "class_name_zh": checkpoint["classes_zh"][class_id],
        "confidence": confidence, "threshold": scene["threshold"],
        "requires_review": bool(requires_review), "action_zh": "需人工复核" if requires_review else "自动分类",
        "checkpoint_sha256": digest, "input_scope": "已裁剪的单个交通标志图像",
    }
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    if args.output:
        save_json(args.output, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "classify"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--result-dir", default=str(ROOT / "results"))
        subparser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
        subparser.add_argument("--threads", type=int, default=4)
        if command == "evaluate":
            subparser.add_argument("--data-dir", default=str(ROOT / "data"))
            subparser.add_argument("--batch-size", type=int, default=256)
        else:
            subparser.add_argument("image")
            subparser.add_argument("--output", help="Optional JSON prediction output path")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_scene(args)
    else:
        classify_image(args)
