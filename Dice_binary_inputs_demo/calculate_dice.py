"""Compute sample-level Dice from strictly aligned binary spot tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['sample', 'spot_id']
COLUMNS = KEYS + ['is_hypo']
CUTOFFS = [30, 35, 40, 45, 50]


def validate_labels(frame, name='labels'):
    if list(frame.columns) != COLUMNS:
        raise ValueError(f'{name}: expected exactly {COLUMNS}')
    if frame.empty or frame.isna().any().any():
        raise ValueError(f'{name}: empty table or missing values')
    if any(frame[key].astype(str).str.strip().eq('').any() for key in KEYS):
        raise ValueError(f'{name}: blank sample or spot ID')
    if frame.duplicated(KEYS).any():
        raise ValueError(f'{name}: duplicate sample/spot IDs')
    if not frame.is_hypo.isin([0, 1]).all():
        raise ValueError(f'{name}: is_hypo must contain only 0 and 1')


def read_labels(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f'{path}: expected exactly {COLUMNS}')
    if not frame.is_hypo.isin(['0', '1']).all():
        raise ValueError(f'{path}: is_hypo must be written as 0 or 1')
    frame['is_hypo'] = frame.is_hypo.astype('uint8')
    validate_labels(frame, str(path))
    return frame


def align_labels(reference, prediction):
    validate_labels(reference, 'reference')
    validate_labels(prediction, 'prediction')
    ref = reference.set_index(KEYS)['is_hypo'].sort_index()
    pred = prediction.set_index(KEYS)['is_hypo'].sort_index()
    missing = ref.index.difference(pred.index)
    extra = pred.index.difference(ref.index)
    if len(missing) or len(extra):
        raise ValueError(f'Spot sets differ: {len(missing)} missing and {len(extra)} extra prediction rows')
    return pd.DataFrame({'mask_is_hypo': ref, 'prediction_is_hypo': pred.reindex(ref.index)}).reset_index()


def count_binary_pair(reference, prediction):
    a, b = np.asarray(reference), np.asarray(prediction)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ValueError('Expected equal non-empty one-dimensional vectors')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Expected binary vectors')
    a, b = a.astype(bool), b.astype(bool)
    tp = int((a & b).sum())
    fp = int((~a & b).sum())
    fn = int((a & ~b).sum())
    tn = int((~a & ~b).sum())
    denominator = int(a.sum() + b.sum())
    return {'n_spots': len(a), 'n_mask_hypo': int(a.sum()), 'n_prediction_hypo': int(b.sum()),
            'intersection': tp, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'denominator': denominator, 'dice': 2 * tp / denominator if denominator else np.nan,
            'dice_defined': denominator > 0}


def calculate_sample_dice(aligned, cutoff):
    rows = []
    for sample, group in aligned.groupby('sample', sort=True):
        counts = count_binary_pair(group.mask_is_hypo, group.prediction_is_hypo)
        rows.append({'sample': sample, 'cutoff_percent': int(cutoff), **counts})
    return pd.DataFrame(rows)


def summarize(per_sample):
    return per_sample.groupby('cutoff_percent', as_index=False, sort=True).agg(
        n_samples=('sample', 'size'), n_defined=('dice', 'count'), mean_dice=('dice', 'mean'),
        sd_dice=('dice', 'std'), min_dice=('dice', 'min'), max_dice=('dice', 'max'))


def save_results(per_sample, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(per_sample)
    wide = per_sample.pivot(index='sample', columns='cutoff_percent', values='dice')
    wide.loc['Equal_sample_mean'] = wide.mean(axis=0)
    per_sample.to_csv(output_dir / 'dice_per_sample.csv', index=False)
    summary.to_csv(output_dir / 'dice_summary.csv', index=False)
    wide.to_csv(output_dir / 'dice_wide.csv')
    return summary, wide


def run(input_dir, output_dir):
    input_dir = Path(input_dir)
    reference = read_labels(input_dir / 'mask_hypo.csv')
    per_sample = pd.concat([
        calculate_sample_dice(align_labels(reference, read_labels(input_dir / f'hypo_{cutoff}.csv')), cutoff)
        for cutoff in CUTOFFS], ignore_index=True)
    summary, wide = save_results(per_sample, output_dir)
    return per_sample, summary, wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent / 'inputs')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    _, summary, _ = run(args.input_dir, args.output_dir)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.9f}'))


if __name__ == '__main__':
    main()
