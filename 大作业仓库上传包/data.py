"""Download, verify and prepare GTSRB with a track-disjoint validation split.

Only the official training split is divided into training and validation.
Images are resized as complete RGB crops; the annotation ROI is not recropped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": "d32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c",
    "GTSRB_Final_Test_Images.zip": "48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa",
    "GTSRB_Final_Test_GT.zip": "f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d",
}
CLASS_NAMES = [
    "Speed limit 20 km/h", "Speed limit 30 km/h", "Speed limit 50 km/h",
    "Speed limit 60 km/h", "Speed limit 70 km/h", "Speed limit 80 km/h",
    "End of speed limit 80 km/h", "Speed limit 100 km/h", "Speed limit 120 km/h",
    "No passing", "No passing for vehicles over 3.5 t",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 t prohibited", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing for vehicles over 3.5 t",
]
CLASS_NAMES_ZH = [
    "限速20公里/小时", "限速30公里/小时", "限速50公里/小时", "限速60公里/小时",
    "限速70公里/小时", "限速80公里/小时", "解除限速80公里/小时", "限速100公里/小时",
    "限速120公里/小时", "禁止超车", "禁止3.5吨以上车辆超车", "下一路口优先通行",
    "优先道路", "减速让行", "停车让行", "禁止车辆通行", "禁止3.5吨以上车辆通行",
    "禁止驶入", "注意危险", "向左急弯", "向右急弯", "连续弯路", "路面不平",
    "易滑路面", "右侧变窄", "道路施工", "交通信号灯", "注意行人", "注意儿童",
    "注意自行车", "注意冰雪", "注意野生动物", "解除限速及超车限制", "前方右转",
    "前方左转", "只准直行", "直行或右转", "直行或左转", "靠右行驶", "靠左行驶",
    "环岛行驶", "解除禁止超车", "解除3.5吨以上车辆禁止超车",
]
CACHE_VERSION = 1


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_one(item: tuple[str, str], archives_dir: Path) -> dict:
    name, expected_sha256 = item
    target = archives_dir / name
    if not target.exists() or file_hash(target) != expected_sha256:
        partial = target.with_suffix(target.suffix + ".part")
        url = BASE_URL + name
        for attempt in range(3):
            try:
                print(f"Downloading {name} (attempt {attempt + 1})", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})
                with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if file_hash(partial) != expected_sha256:
                    raise RuntimeError(f"SHA256 verification failed for {name}")
                partial.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    result = {"filename": name, "url": BASE_URL + name,
              "bytes": target.stat().st_size, "md5": file_hash(target, "md5"),
              "sha256": file_hash(target), "expected_sha256": expected_sha256,
              "checksum_reference": "https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/"}
    assert result["sha256"] == expected_sha256
    print(f"Verified {name}: {result['bytes']} bytes", flush=True)
    return result


def _safe_extract(archive: Path, destination: Path) -> None:
    """Reject paths or links that could write outside the destination."""
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            candidate = (root / member.filename).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlinks are unsupported: {member.filename}")
        zf.extractall(root)


def _get_records(raw: Path) -> tuple[list[dict], list[dict]]:
    train_root = raw / "GTSRB" / "Final_Training" / "Images"
    train = []
    for class_id in range(43):
        class_dir = train_root / f"{class_id:05d}"
        for path in sorted(class_dir.glob("*.ppm")):
            train.append({"path": path.relative_to(raw).as_posix(), "class_id": class_id,
                          "track": f"{class_id:05d}/{path.stem.split('_')[0]}"})
    test_root = raw / "GTSRB" / "Final_Test" / "Images"
    with (raw / "GT-final_test.csv").open(encoding="utf-8-sig", newline="") as stream:
        test = [{"path": (test_root / row["Filename"]).relative_to(raw).as_posix(),
                 "class_id": int(row["ClassId"])}
                for row in csv.DictReader(stream, delimiter=";")]
    test.sort(key=lambda x: x["path"])
    if len(train) != 39209 or len(test) != 12630:
        raise ValueError(f"Unexpected official split counts: {len(train)}, {len(test)}")
    if len({row["path"] for row in test}) != len(test):
        raise ValueError("Duplicated official test filename")
    assert {r["class_id"] for r in train} == set(range(43))
    assert {r["class_id"] for r in test} == set(range(43))
    return train, test


def _split_tracks(records: list[dict], seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    val_tracks = set()
    for class_id in range(43):
        tracks = sorted({r["track"] for r in records if r["class_id"] == class_id})
        # Round to the nearest whole physical sign, retaining both splits per class.
        n_val = max(1, min(len(tracks) - 1, int(len(tracks) * val_fraction + 0.5)))
        val_tracks.update(rng.permutation(tracks)[:n_val].tolist())
    val_indices = np.array([i for i, r in enumerate(records) if r["track"] in val_tracks], dtype=np.int64)
    train_indices = np.array([i for i, r in enumerate(records) if r["track"] not in val_tracks], dtype=np.int64)
    train_tracks = {records[i]["track"] for i in train_indices}
    assert train_tracks.isdisjoint(val_tracks)
    assert len(train_indices) + len(val_indices) == len(records)
    class_counts = []
    for class_id in range(43):
        train_rows = [records[i] for i in train_indices if records[i]["class_id"] == class_id]
        val_rows = [records[i] for i in val_indices if records[i]["class_id"] == class_id]
        class_counts.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id],
                             "train_images": len(train_rows), "val_images": len(val_rows),
                             "train_tracks": len({r["track"] for r in train_rows}),
                             "val_tracks": len({r["track"] for r in val_rows})})
    return train_indices, val_indices, class_counts


def _image_cache(records: list[dict], raw: Path, output: Path, image_size: int) -> None:
    array = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                     shape=(len(records), 3, image_size, image_size))
    for i, row in enumerate(records):
        with Image.open(raw / row["path"]) as im:
            image = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            array[i] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        if (i + 1) % 10000 == 0:
            print(f"Cached {i + 1}/{len(records)} images", flush=True)
    array.flush()
    del array


def _training_statistics(images: np.ndarray, train_indices: np.ndarray) -> tuple[list[float], list[float]]:
    total = np.zeros(3, dtype=np.float64)
    square = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), 256):
        batch = images[train_indices[start:start + 256]].astype(np.float64) / 255.0
        total += batch.sum(axis=(0, 2, 3))
        square += np.square(batch).sum(axis=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 0))
    assert np.all(std > 0)
    return mean.tolist(), std.tolist()


def prepare_data(data_dir: str | Path = "data", image_size: int = 64, seed: int = 42,
                 val_fraction: float = 0.2, download: bool = True) -> Path:
    """Prepare a cache once; return its directory. No stochastic augmentation."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    root = Path(data_dir)
    fraction_tag = str(val_fraction).replace(".", "p")
    cache = root / f"cache_{image_size}_seed{seed}_val{fraction_tag}"
    metadata_path = cache / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_version") != CACHE_VERSION:
            raise ValueError("Cache version changed; use a fresh data directory")
        for name, expected in metadata["cache_sha256"].items():
            path = cache / name
            if not path.exists() or file_hash(path) != expected:
                raise ValueError(f"Cache verification failed: {path}")
        if "split_digest" not in metadata:
            metadata["split_digest"] = file_hash(cache / "split_manifest.json")
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return cache
    archives_dir = root / "archives"
    raw = root / "raw"
    archives_dir.mkdir(parents=True, exist_ok=True)
    if download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            archives = list(pool.map(lambda item: _download_one(item, archives_dir), ARCHIVES.items()))
    else:
        archives = []
        for name, expected_sha256 in ARCHIVES.items():
            path = archives_dir / name
            if not path.exists() or file_hash(path) != expected_sha256:
                raise ValueError(f"Missing or invalid local archive {path}")
            archives.append({"filename": name, "url": BASE_URL + name, "md5": file_hash(path, "md5"),
                             "sha256": file_hash(path), "bytes": path.stat().st_size})
    for archive in archives:
        _safe_extract(archives_dir / archive["filename"], raw)
    train, test = _get_records(raw)
    train_indices, val_indices, counts = _split_tracks(train, seed, val_fraction)
    cache.mkdir(parents=True, exist_ok=True)
    _image_cache(train, raw, cache / "official_train_images.npy", image_size)
    _image_cache(test, raw, cache / "test_images.npy", image_size)
    np.save(cache / "official_train_labels.npy", np.array([r["class_id"] for r in train], dtype=np.int64))
    np.save(cache / "test_labels.npy", np.array([r["class_id"] for r in test], dtype=np.int64))
    np.save(cache / "train_indices.npy", train_indices)
    np.save(cache / "val_indices.npy", val_indices)
    image_array = np.load(cache / "official_train_images.npy", mmap_mode="r")
    mean, std = _training_statistics(image_array, train_indices)
    records = {"official_train": train, "test": test,
               "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist()}
    (cache / "split_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in counts:
        entry["test_images"] = sum(r["class_id"] == entry["class_id"] for r in test)
    metadata = {
        "cache_version": CACHE_VERSION, "dataset": "GTSRB", "classes": CLASS_NAMES,
        "classes_zh": CLASS_NAMES_ZH, "image_size": image_size, "seed": seed,
        "requested_val_fraction": val_fraction,
        "split_strategy": "Per-class random physical-track split, full official test unchanged",
        "train_count": len(train_indices), "val_count": len(val_indices), "test_count": len(test),
        "official_train_count": len(train),
        "train_track_count": sum(r["train_tracks"] for r in counts),
        "val_track_count": sum(r["val_tracks"] for r in counts), "track_overlap": 0,
        "split_digest": file_hash(cache / "split_manifest.json"),
        "mean": mean, "std": std, "statistics_source": "training portion after resizing, pixels / 255",
        "preprocessing": "Full-image RGB conversion and Pillow bilinear resize, no ROI recrop",
        "array_layout": "NCHW uint8", "class_counts": counts, "archives": archives,
        "sources": ["https://github.com/houbensebastian/GermanTrafficSignBenchmarks",
                    "https://galilai-group.github.io/stable-datasets/_modules/stable_datasets/images/gtsrb/"],
        "cache_sha256": {p.name: file_hash(p) for p in sorted(cache.iterdir()) if p.is_file()},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def load_data(data_dir: str | Path = "data", image_size: int = 64, seed: int = 42,
              val_fraction: float = 0.2, download: bool = True) -> dict:
    """Return NCHW uint8 images, int64 labels, training-only statistics and metadata."""
    cache = prepare_data(data_dir, image_size, seed, val_fraction, download)
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache / "split_manifest.json").read_text(encoding="utf-8"))
    official_images = np.load(cache / "official_train_images.npy", mmap_mode="r")
    official_labels = np.load(cache / "official_train_labels.npy")
    train_indices = np.load(cache / "train_indices.npy")
    val_indices = np.load(cache / "val_indices.npy")
    result = {"train_images": official_images[train_indices],
              "val_images": official_images[val_indices],
              "test_images": np.load(cache / "test_images.npy", mmap_mode="r"),
              "train_labels": official_labels[train_indices], "val_labels": official_labels[val_indices],
              "test_labels": np.load(cache / "test_labels.npy"), "mean": metadata["mean"],
              "std": metadata["std"], "metadata": metadata,
              "classes": CLASS_NAMES.copy(), "classes_zh": CLASS_NAMES_ZH.copy(),
              "train_paths": [manifest["official_train"][i]["path"] for i in train_indices],
              "val_paths": [manifest["official_train"][i]["path"] for i in val_indices],
              "test_paths": [r["path"] for r in manifest["test"]], "cache_dir": str(cache)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    prepared = prepare_data(args.data_dir, args.image_size, args.seed, args.val_fraction, not args.no_download)
    info = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps({k: info[k] for k in ["train_count", "val_count", "test_count", "train_track_count",
                                         "val_track_count", "track_overlap", "mean", "std"]}, indent=2))
    print(f"Prepared cache: {prepared}")
