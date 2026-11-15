"""Check that saved checkpoints, predictions, metrics and source hashes agree."""
import json
from pathlib import Path

import numpy as np
import torch

from train import classification_metrics, save_json, sha256

ROOT = Path(__file__).resolve().parent


def selective_from_saved(probabilities, labels, threshold, reject_all=False):
    """Recompute acceptance counts directly from recorded probabilities."""
    accepted = np.zeros(len(labels), dtype=bool) if reject_all else probabilities.max(1) >= threshold
    count = int(accepted.sum())
    correct = int(((probabilities.argmax(1) == labels) & accepted).sum())
    return {'count': int(len(labels)), 'accepted': count, 'rejected': int(len(labels) - count),
            'coverage': count / len(labels), 'accepted_correct': correct,
            'accepted_errors': count - correct,
            'accuracy_on_accepted': correct / count if count else None}


def main():
    results = ROOT / 'results'
    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    scene = json.loads((results / 'scene.json').read_text(encoding='utf-8'))
    assert set(summary['models']) == {'baseline', 'se', 'depthwise', 'depthwise_se'}
    for name, expected in summary['provenance']['source_sha256'].items():
        assert sha256(ROOT / name) == expected, f'Training source changed: {name}'
    assert scene['summary_sha256'] == sha256(results / 'summary.json')
    assert summary['data']['track_overlap'] == 0
    assert summary['data']['train_count'] + summary['data']['val_count'] == 39209
    assert summary['data']['test_count'] == 12630
    # Verify actual track membership and label order, not just reported counts.
    cache = ROOT / 'data' / f"cache_{summary['config']['image_size']}_seed{summary['config']['split_seed']}_val0p2"
    manifest_path = cache / 'split_manifest.json'
    assert sha256(manifest_path) == summary['data']['split_digest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    train_indices = np.load(cache / 'train_indices.npy', allow_pickle=False)
    val_indices = np.load(cache / 'val_indices.npy', allow_pickle=False)
    assert train_indices.tolist() == manifest['train_indices']
    assert val_indices.tolist() == manifest['val_indices']
    assert np.array_equal(np.sort(np.concatenate([train_indices, val_indices])), np.arange(39209))
    train_tracks = {manifest['official_train'][int(i)]['track'] for i in train_indices}
    val_tracks = {manifest['official_train'][int(i)]['track'] for i in val_indices}
    assert train_tracks.isdisjoint(val_tracks)
    official_labels = np.load(cache / 'official_train_labels.npy', allow_pickle=False)
    assert official_labels.tolist() == [r['class_id'] for r in manifest['official_train']]
    expected_labels = {'val': official_labels[val_indices],
                       'test': np.load(cache / 'test_labels.npy', allow_pickle=False)}
    assert expected_labels['test'].tolist() == [r['class_id'] for r in manifest['test']]
    for name, expected in scene['protocol']['source_sha256'].items():
        assert sha256(ROOT / name) == expected, f'Scene source changed: {name}'
    chosen = min(['depthwise', 'depthwise_se'], key=lambda name: (
        -summary['models'][name]['validation']['accuracy'], summary['models'][name]['profile']['parameters']))
    assert scene['chosen_variant'] == chosen
    assert scene['selection']['uses_test_for_selection'] is False
    checked = {}
    for name, model in summary['models'].items():
        path = ROOT / 'checkpoints' / (name + '.pt')
        assert sha256(path) == model['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['source_sha256'] == summary['provenance']['source_sha256']
        assert checkpoint['split_digest'] == summary['data']['split_digest']
        assert checkpoint['epoch'] == model['best_epoch']
        assert checkpoint['training_config'] == summary['config'] == model['config']
        assert model['provenance'] == summary['provenance']
        assert len(model['history']) == summary['config']['epochs']
        best = max(row['val_accuracy'] for row in model['history'])
        earliest = next(row['epoch'] for row in model['history'] if row['val_accuracy'] == best)
        assert earliest == model['best_epoch']
        assert best == model['validation']['accuracy']
        with np.load(results / (name + '_predictions.npz'), allow_pickle=False) as saved:
            for split in ['val', 'test']:
                probs = saved[split + '_probabilities']
                labels = saved[split + '_labels']
                assert np.array_equal(labels, expected_labels[split]), (name, split, 'label order')
                assert probs.shape == (len(labels), 43)
                assert np.all((probs >= 0) & (probs <= 1))
                assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1, atol=1e-5)
                metrics = classification_metrics(labels, probs.argmax(1))
                reference = model['validation' if split == 'val' else 'test']
                for key in ['count', 'correct', 'accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix', 'per_class']:
                    assert metrics[key] == reference[key], (name, split, key)
            if name == chosen:
                val_probs, val_labels = saved['val_probabilities'], saved['val_labels']
                grid = []
                for threshold in scene['selection']['threshold_grid']:
                    row = {'threshold': threshold, **selective_from_saved(val_probs, val_labels, threshold)}
                    row['meets_target'] = row['accepted'] > 0 and row['accepted_correct'] * 100 >= row['accepted'] * 99
                    grid.append(row)
                assert grid == scene['selection']['validation_grid_results']
                eligible = [row for row in grid if row['meets_target']]
                optimum = min(eligible, key=lambda row: (-row['accepted'], row['threshold'])) if eligible else None
                assert scene['threshold'] == (optimum['threshold'] if optimum else None)
                assert scene['reject_all'] == (optimum is None)
                assert selective_from_saved(val_probs, val_labels, scene['threshold'], scene['reject_all']) == scene['application']['validation']
                assert selective_from_saved(saved['test_probabilities'], saved['test_labels'], scene['threshold'], scene['reject_all']) == scene['application']['test_conditions']['clean']
                assert scene['application']['checkpoint_sha256'] == model['checkpoint_sha256']
        assert scene['models'][name]['clean_correct_matches_training']
        assert scene['models'][name]['checkpoint_sha256'] == model['checkpoint_sha256']
        assert scene['models'][name]['conditions']['clean']['correct'] == model['test']['correct']
        assert abs(scene['models'][name]['conditions']['clean']['macro_f1'] - model['test']['macro_f1']) < 1e-12
        clean = scene['models'][name]['conditions']['clean']['accuracy']
        for condition, metrics in scene['models'][name]['conditions'].items():
            assert metrics['count'] == 12630
            assert metrics['accuracy'] == metrics['correct'] / 12630
            assert abs(metrics['decline_pp'] - (clean - metrics['accuracy']) * 100) < 1e-10
        checked[name] = {'best_epoch': model['best_epoch'], 'test_correct': model['test']['correct'],
                         'test_accuracy': model['test']['accuracy'], 'checkpoint_sha256': model['checkpoint_sha256']}
    for group in [scene['application']['validation'], *scene['application']['test_conditions'].values()]:
        assert group['accepted'] + group['rejected'] == group['count']
        assert group['accepted_correct'] + group['accepted_errors'] == group['accepted']
        assert group['coverage'] == group['accepted'] / group['count']
        if group['accepted']:
            assert group['accuracy_on_accepted'] == group['accepted_correct'] / group['accepted']
        else:
            assert group['accuracy_on_accepted'] is None
    prediction = json.loads((results / 'example_prediction.json').read_text(encoding='utf-8'))
    assert prediction['class_id'] == scene['example']['predicted_class_id']
    assert prediction['requires_review'] == scene['example']['requires_review']
    assert prediction['variant'] == chosen
    assert prediction['threshold'] == scene['threshold']
    assert prediction['checkpoint_sha256'] == scene['application']['checkpoint_sha256']
    result = {'status': 'passed', 'source_hashes_match_training': True,
              'track_overlap': 0, 'models': checked, 'scene_and_predictions_consistent': True,
              'validation_only_model_and_threshold_selection_verified': True,
              'saved_labels_match_split_manifest': True,
              'cpu_example_class_matches_gpu': True,
              'baseline_project_target_met': summary['models']['baseline']['test']['accuracy'] >= 0.95}
    save_json(results / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
