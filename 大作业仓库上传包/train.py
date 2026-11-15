"""Train a controlled 2x2 ResNet-18 ablation on track-disjoint GTSRB splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from data import load_data
from models import build_model, profile_model

ROOT = Path(__file__).resolve().parent
VARIANTS = ['baseline', 'se', 'depthwise', 'depthwise_se']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def configure(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def preprocess(images, mean, std, augmentation_generator=None):
    x = images.to(torch.float32).div_(255)
    if augmentation_generator is not None:
        # A separate RNG gives every architecture identical per-batch transforms.
        b = len(x)
        r = torch.rand((b, 5), device=x.device, generator=augmentation_generator)
        angles = (r[:, 0] * 2 - 1) * (10 * math.pi / 180)
        scale = 0.9 + r[:, 1] * 0.2
        theta = torch.zeros((b, 2, 3), device=x.device)
        theta[:, 0, 0] = angles.cos() / scale
        theta[:, 0, 1] = -angles.sin() / scale
        theta[:, 1, 0] = angles.sin() / scale
        theta[:, 1, 1] = angles.cos() / scale
        theta[:, :, 2] = (r[:, 2:4] * 2 - 1) * 0.12
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        x = (x * (0.8 + 0.4 * r[:, 4, None, None, None])).clamp_(0, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def classification_metrics(labels, predictions, num_classes=43):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    cm = np.bincount(labels * num_classes + predictions, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(num_classes, dtype=float), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(num_classes, dtype=float), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes, dtype=float), where=precision + recall != 0)
    return {'count': int(len(labels)), 'correct': int(tp.sum()), 'accuracy': float(tp.sum() / len(labels)),
            'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()),
            'confusion_matrix': cm.tolist(), 'per_class': [
                {'class_id': i, 'support': int(cm[i].sum()), 'correct': int(tp[i]),
                 'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                for i in range(num_classes)]}


@torch.inference_mode()
def evaluate(model, images, labels, mean, std, batch_size=256, return_probabilities=False, transform=None):
    model.eval()
    all_probabilities = []
    loss_sum = 0.0
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        if transform is None:
            x = preprocess(chunk, mean, std)
        else:
            x = transform(chunk.float() / 255)
            x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
        # Evaluation is FP32 for all models and corruption conditions.
        logits = model(x)
        target = labels[start:start + batch_size]
        loss_sum += F.cross_entropy(logits, target, reduction='sum').item()
        all_probabilities.append(logits.softmax(1).cpu().numpy())
    probs = np.concatenate(all_probabilities)
    metrics = classification_metrics(labels.cpu().numpy(), probs.argmax(1))
    metrics['cross_entropy'] = loss_sum / len(images)
    return (metrics, probs) if return_probabilities else metrics


def checkpoint_cpu(model, variant, data, config, epoch, val_accuracy):
    return {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'variant': variant, 'num_classes': 43, 'input_size': config['image_size'],
            'mean': data['mean'], 'std': data['std'], 'classes': data['classes'],
            'classes_zh': data['classes_zh'], 'epoch': epoch, 'validation_accuracy': val_accuracy,
            'training_config': config, 'split_digest': data['metadata'].get('split_digest'),
            'source_sha256': {name: sha256(ROOT / name) for name in ['models.py', 'data.py', 'train.py']}}


def run(args):
    configure(args.seed, args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly.')
    result_dir, checkpoint_dir = Path(args.result_dir), Path(args.checkpoint_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {'seed': args.seed, 'split_seed': 42, 'image_size': 64, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size,
              'optimizer': 'AdamW', 'learning_rate': args.lr, 'weight_decay': 1e-4,
              'loss': 'cross_entropy', 'label_smoothing': 0.0, 'warmup_epochs': 2,
              'minimum_lr_ratio': 0.02, 'threads': args.threads,
              'augmentation': {'rotation_degrees': 10, 'translation_fraction': 0.06,
                               'scale': [0.9, 1.1], 'brightness': [0.8, 1.2],
                               'horizontal_flip': False, 'vertical_flip': False,
                               'padding': 'border', 'interpolation': 'bilinear'},
              'selection': 'maximum validation accuracy; earliest epoch on ties',
              'baseline_project_target_accuracy': 0.95, 'pretrained': False,
              'training_precision': 'AMP float16' if device.type == 'cuda' else 'FP32',
              'evaluation_precision': 'FP32'}
    data = load_data(args.data_dir, image_size=64, seed=42, val_fraction=0.2)
    provenance = {'split_digest': data['metadata']['split_digest'],
                  'source_sha256': {name: sha256(ROOT / name) for name in ['models.py', 'data.py', 'train.py']}}
    mean = torch.tensor(data['mean'], device=device)[None, :, None, None]
    std = torch.tensor(data['std'], device=device)[None, :, None, None]
    splits = {}
    for split in ['train', 'val', 'test']:
        # Keep only uint8 image data resident on GPU; normalize one batch at a time.
        splits[split] = (torch.as_tensor(np.ascontiguousarray(data[split + '_images']), device=device),
                         torch.as_tensor(data[split + '_labels'], dtype=torch.long, device=device))
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'numpy': np.__version__,
                   'platform': platform.platform(), 'device': str(device),
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'cuda_runtime': torch.version.cuda,
                   'cudnn_version': torch.backends.cudnn.version(),
                   'cudnn_deterministic': True, 'tf32': False}
    save_json(result_dir / 'experiment_config.json', {'config': config, 'environment': environment,
              'data': data['metadata'], 'classes': data['classes'], 'classes_zh': data['classes_zh']})
    print(json.dumps({'event': 'start', 'counts': {s: len(v[0]) for s, v in splits.items()},
                      'config': config, 'environment': environment}), flush=True)
    summary = {'config': config, 'environment': environment, 'data': data['metadata'],
               'classes': data['classes'], 'classes_zh': data['classes_zh'], 'models': {}, 'provenance': provenance}
    for variant in args.variants:
        configure(args.seed, args.threads)
        result_path = result_dir / (variant + '.json')
        best_path = checkpoint_dir / (variant + '.pt')
        latest_path = checkpoint_dir / (variant + '_last.pt')
        if args.resume and result_path.exists() and best_path.exists():
            done = json.loads(result_path.read_text(encoding='utf-8'))
            if done['config'] == config and done.get('provenance') == provenance and done['checkpoint_sha256'] == sha256(best_path):
                summary['models'][variant] = done
                print(json.dumps({'event': 'already_complete', 'variant': variant}), flush=True)
                continue
        model = build_model(variant, num_classes=43)
        profile = profile_model(model, input_size=(1, 3, 64, 64))
        model = model.to(device=device, memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        history, best_acc, best_epoch, start_epoch = [], -1.0, 0, 1
        if args.resume and latest_path.exists():
            saved = torch.load(latest_path, map_location=device, weights_only=False)
            if saved['config'] != config or saved.get('provenance') != provenance:
                raise ValueError('Resume config, source code or split changed; use a new result/checkpoint directory.')
            if not best_path.is_file():
                raise ValueError('Resume requires the recorded best checkpoint.')
            previous_best = torch.load(best_path, map_location='cpu', weights_only=False)
            if previous_best['epoch'] != saved['best_epoch'] or previous_best['validation_accuracy'] != saved['best_acc']:
                raise ValueError('Resume best checkpoint does not match the last epoch state.')
            model.load_state_dict(saved['state_dict'])
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            history, best_acc, best_epoch = saved['history'], saved['best_acc'], saved['best_epoch']
            start_epoch = saved['epoch'] + 1
        for epoch in range(start_epoch, args.epochs + 1):
            tic = time.perf_counter()
            if epoch <= 2:
                lr = args.lr * epoch / 2
            else:
                progress = (epoch - 2) / max(1, args.epochs - 2)
                lr = args.lr * (0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            sample_rng = torch.Generator().manual_seed(args.seed * 100000 + epoch)
            augment_rng = torch.Generator(device=device).manual_seed(args.seed * 100000 + epoch + 10000)
            order = torch.randperm(len(splits['train'][0]), generator=sample_rng).to(device)
            model.train()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), dtype=torch.long, device=device)
            seen = 0
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                x = preprocess(splits['train'][0][idx], mean, std, augment_rng)
                y = splits['train'][1][idx]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.detach().float() * len(idx)
                correct += (logits.detach().argmax(1) == y).sum()
                seen += len(idx)
            val = evaluate(model, *splits['val'], mean, std, args.eval_batch_size)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            row = {'epoch': epoch, 'learning_rate': lr, 'train_loss': loss_sum.item() / seen,
                   'train_accuracy': correct.item() / seen, 'val_loss': val['cross_entropy'],
                   'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'],
                   'seconds': time.perf_counter() - tic}
            history.append(row)
            if val['accuracy'] > best_acc:
                best_acc, best_epoch = val['accuracy'], epoch
                save_checkpoint(best_path, checkpoint_cpu(model, variant, data, config, epoch, best_acc))
            save_checkpoint(latest_path, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), 'history': history, 'best_acc': best_acc,
                        'best_epoch': best_epoch, 'epoch': epoch, 'config': config, 'provenance': provenance})
            save_json(result_dir / (variant + '_progress.json'), {'variant': variant, 'history': history,
                      'best_epoch': best_epoch, 'best_val_accuracy': best_acc})
            print(json.dumps({'event': 'epoch', 'variant': variant, **row}), flush=True)
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best['state_dict'])
        val_metrics, val_probs = evaluate(model, *splits['val'], mean, std, args.eval_batch_size, True)
        test_metrics, test_probs = evaluate(model, *splits['test'], mean, std, args.eval_batch_size, True)
        np.savez_compressed(result_dir / (variant + '_predictions.npz'),
                            val_probabilities=val_probs, test_probabilities=test_probs,
                            val_labels=data['val_labels'], test_labels=data['test_labels'])
        result = {'variant': variant, 'config': config, 'provenance': provenance, 'profile': profile, 'history': history,
                  'best_epoch': best_epoch, 'validation': val_metrics, 'test': test_metrics,
                  'training_seconds': sum(v['seconds'] for v in history),
                  'checkpoint_sha256': sha256(best_path), 'checkpoint_bytes': best_path.stat().st_size,
                  'baseline_target_met': test_metrics['accuracy'] >= 0.95 if variant == 'baseline' else None}
        save_json(result_path, result)
        summary['models'][variant] = result
        save_json(result_dir / 'summary.json', summary)
        print(json.dumps({'event': 'complete', 'variant': variant, 'best_epoch': best_epoch,
                          'test_accuracy': test_metrics['accuracy'], 'macro_f1': test_metrics['macro_f1'],
                          'profile': profile}), flush=True)
        del model, optimizer, scaler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    save_json(result_dir / 'summary.json', summary)
    with open(result_dir / 'metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'parameters', 'macs', 'best_epoch',
                                            'validation_accuracy', 'test_accuracy', 'macro_f1', 'correct', 'count'])
        writer.writeheader()
        for name, r in summary['models'].items():
            writer.writerow({'variant': name, 'parameters': r['profile']['parameters'],
                             'macs': r['profile']['macs'], 'best_epoch': r['best_epoch'],
                             'validation_accuracy': r['validation']['accuracy'],
                             'test_accuracy': r['test']['accuracy'], 'macro_f1': r['test']['macro_f1'],
                             'correct': r['test']['correct'], 'count': r['test']['count']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--result-dir', default=str(ROOT / 'results'))
    parser.add_argument('--checkpoint-dir', default=str(ROOT / 'checkpoints'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
