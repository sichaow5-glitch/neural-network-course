"""Plot measured training and evaluation outputs without rerunning inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ORDER = ["baseline", "se", "depthwise", "depthwise_se"]
LABELS = {"baseline": "ResNet-18", "se": "ResNet-18 + SE",
          "depthwise": "Depthwise", "depthwise_se": "Depthwise + SE"}
COLORS = {"baseline": "#264b78", "se": "#df8c2e", "depthwise": "#2a8c7c",
          "depthwise_se": "#9460a1"}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def finish(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_lightweight(models):
    candidates = [v for v in ("depthwise", "depthwise_se") if v in models]
    if not candidates:
        candidates = list(models)
    return sorted(candidates, key=lambda v: (-models[v]["validation"]["accuracy"],
                                            models[v]["profile"]["parameters"], v))[0]


def plot_learning(models, directory):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    specs = [("train_loss", "Training cross-entropy", 1),
             ("val_loss", "Validation cross-entropy", 1),
             ("train_accuracy", "Training accuracy (%)", 100),
             ("val_accuracy", "Validation accuracy (%)", 100)]
    for ax, (key, ylabel, scale) in zip(axs.flat, specs):
        for name, r in models.items():
            history = r["history"]
            ax.plot([x["epoch"] for x in history], [x[key] * scale for x in history],
                    color=COLORS[name], label=LABELS[name], linewidth=1.8)
            if key == "val_accuracy":
                best = next(x for x in history if x["epoch"] == r["best_epoch"])
                ax.scatter([best["epoch"]], [best[key] * scale], color=COLORS[name], s=30, zorder=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=9.5, frameon=False)
    finish(fig, directory / "learning_curves.png")


def plot_metrics(models, directory):
    names = list(models)
    short = ["Base" if v == "baseline" else "+ SE" if v == "se" else "DW" if v == "depthwise" else "DW + SE" for v in names]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    series = [("Test accuracy (%)", [models[n]["test"]["accuracy"] * 100 for n in names], ".2f"),
              ("Parameters (millions)", [models[n]["profile"]["parameters"] / 1e6 for n in names], ".3f"),
              ("Conv / Linear MACs (millions)", [models[n]["profile"]["macs"] / 1e6 for n in names], ".2f")]
    for ax, (title, values, fmt) in zip(axs, series):
        bars = ax.bar(short, values, color=[COLORS[n] for n in names], width=0.65)
        ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=10)
        ax.set_ylim(0, max(values) * 1.14)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, directory / "metrics_comparison.png")


def plot_confusion(result, classes, directory):
    cm = np.asarray(result["test"]["confusion_matrix"], dtype=np.int64)
    support = cm.sum(1)
    norm = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float), where=support[:, None] != 0)
    fig, ax = plt.subplots(figsize=(10.8, 9.4), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(cm)), labels=np.arange(len(cm)), fontsize=8)
    ax.set_yticks(np.arange(len(cm)), labels=np.arange(len(cm)), fontsize=8)
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ax.set_title(LABELS[result["variant"]] + " | row-normalized test confusion matrix", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, label="Fraction within true class")
    finish(fig, directory / "confusion_matrix.png")

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    pairs = [(int(i), int(j), int(off_diag[i, j])) for i, j in zip(*np.nonzero(off_diag))]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    shown = pairs[:10]
    fig, ax = plt.subplots(figsize=(10.8, max(3.4, len(shown) * 0.45 + 1.3)), constrained_layout=True)
    if shown:
        labels = [f"{i}: {classes[i]}  →  {j}: {classes[j]}" for i, j, _ in shown]
        bars = ax.barh(np.arange(len(shown)), [x[2] for x in shown], color=COLORS[result["variant"]])
        ax.set_yticks(np.arange(len(shown)), labels=labels, fontsize=10)
        ax.invert_yaxis()
        ax.bar_label(bars, padding=4)
        ax.set_xlim(0, max(x[2] for x in shown) * 1.16)
    else:
        ax.text(0.5, 0.5, "No misclassified test samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel("Misclassified test images")
    ax.set_title("Most frequent directed confusion pairs | " + LABELS[result["variant"]])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    finish(fig, directory / "confusion_pairs.png")
    return {"confusion_matrix": cm.tolist(), "top_confusion_pairs": [
        {"true_class": i, "predicted_class": j, "count": count,
         "true_name": classes[i], "predicted_name": classes[j]} for i, j, count in shown]}


def plot_errors(chosen, summary, directory, data_dir):
    from data import load_data
    prediction_path = directory / f"{chosen}_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as predictions:
        probabilities = predictions["test_probabilities"]
        truth = predictions["test_labels"].astype(np.int64)
    pred = probabilities.argmax(1)
    indices = np.flatnonzero(pred != truth)[:8]
    model_result = summary["models"][chosen]
    assert len(truth) == model_result["test"]["count"]
    assert int((pred == truth).sum()) == model_result["test"]["correct"]
    classes = summary["classes"]
    fig, axs = plt.subplots(2, 4, figsize=(11.8, 6.6), constrained_layout=True)
    rows = []
    if len(indices):
        dataset = load_data(data_dir, image_size=summary["config"]["image_size"],
                            seed=summary["config"]["split_seed"], val_fraction=0.2, download=False)
        assert np.array_equal(truth, dataset["test_labels"]), "Test order differs from saved predictions"
        assert dataset["metadata"]["split_digest"] == summary["data"]["split_digest"]
        for ax, index in zip(axs.flat, indices):
            i, j = int(truth[index]), int(pred[index])
            confidence = float(probabilities[index, j])
            ax.imshow(dataset["test_images"][index].transpose(1, 2, 0))
            title = f"Test #{index}\nT {i}: {classes[i]}\nP {j}: {classes[j]}\nScore {confidence:.3f}"
            wrapped = "\n".join(textwrap.fill(line, width=26) for line in title.split("\n"))
            ax.set_title(wrapped, fontsize=9.5, pad=7)
            rows.append({"index": int(index), "path": dataset["test_paths"][index],
                         "true_class": i, "predicted_class": j, "confidence": confidence})
    else:
        axs[0, 0].text(0.5, 0.5, "No test errors", ha="center", va="center")
    for ax in axs.flat:
        ax.axis("off")
    fig.suptitle("First eight errors in official test order | " + LABELS[chosen], fontsize=13)
    finish(fig, directory / "test_errors.png")
    return rows


def plot_robustness(scene, directory):
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    specifications = [([0.5, 1.0, 1.5], ["brightness_0.5", "clean", "brightness_1.5"], "Brightness multiplier"),
                      ([0, 1, 2], ["clean", "blur_sigma1", "blur_sigma2"], "Gaussian blur sigma (pixels)")]
    for ax, (x_values, conditions, xlabel) in zip(axs, specifications):
        for variant in ORDER:
            if variant not in scene["models"]:
                continue
            data = scene["models"][variant]["conditions"]
            if not all(c in data for c in conditions):
                raise ValueError(f"Missing robustness conditions for {variant}: {conditions}")
            y = [data[c]["accuracy"] * 100 for c in conditions]
            ax.plot(x_values, y, marker="o", color=COLORS[variant], label=LABELS[variant], linewidth=1.8)
        ax.set_xticks(x_values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 100.8)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=9, loc="lower left", frameon=False)
    finish(fig, directory / "robustness_curves.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(ROOT / "results"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--skip-error-images", action="store_true", help="Plot metrics without loading local image cache")
    args = parser.parse_args()
    directory = Path(args.result_dir)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    models = {name: summary["models"][name] for name in ORDER if name in summary["models"]}
    if not models:
        raise ValueError("No completed models in summary.json")
    scene_path = directory / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.exists() else None
    chosen = scene["chosen_variant"] if scene else select_lightweight(models)
    if chosen not in models:
        raise ValueError("Chosen scene model is absent from summary.json")
    if scene:
        assert scene['summary_sha256'] == hashlib.sha256((directory / 'summary.json').read_bytes()).hexdigest(), 'Scene refers to an older training summary'
        assert chosen == select_lightweight(models), 'Scene model differs from validation-only selection'
        for name, r in models.items():
            clean = scene["models"][name]["conditions"]["clean"]
            if clean["count"] != r["test"]["count"] or clean["correct"] != r["test"]["correct"]:
                raise ValueError(f"scene.json and summary.json describe different test outputs for {name}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 10, "savefig.facecolor": "white"})
    plot_learning(models, directory)
    plot_metrics(models, directory)
    details = plot_confusion(models[chosen], summary["classes"], directory)
    details.update({"chosen_variant": chosen,
                    "selection": scene["selection"] if scene else "highest validation accuracy among lightweight variants; ties use fewer parameters",
                    "error_selection": "first eight errors in official test order"})
    if not args.skip_error_images:
        details["error_examples"] = plot_errors(chosen, summary, directory, args.data_dir)
    if scene:
        plot_robustness(scene, directory)
    save_json(directory / "plot_data.json", details)
    print(json.dumps({"chosen_variant": chosen, "results": str(directory),
                      "robustness_plot": scene is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
