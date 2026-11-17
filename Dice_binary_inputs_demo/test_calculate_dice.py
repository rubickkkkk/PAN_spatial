"""Unit and packaged-input tests for binary Dice calculations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from calculate_dice import (
    CUTOFFS, align_labels, calculate_sample_dice, count_binary_pair, read_labels, run, summarize,
)


def table(values, ids=None):
    return pd.DataFrame({'sample': ['S1'] * len(values),
                         'spot_id': ids if ids is not None else [f's{i}' for i in range(len(values))],
                         'is_hypo': values})


class DiceTests(unittest.TestCase):
    def test_counts_and_formula(self):
        got = count_binary_pair([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((got['tp'], got['fp'], got['fn'], got['tn']), (1, 1, 1, 1))
        self.assertEqual(got['dice'], .5)

    def test_perfect_disjoint_and_empty(self):
        self.assertEqual(count_binary_pair([1, 0], [1, 0])['dice'], 1)
        self.assertEqual(count_binary_pair([1, 0], [0, 1])['dice'], 0)
        self.assertEqual(count_binary_pair([1, 0], [0, 0])['dice'], 0)
        self.assertTrue(np.isnan(count_binary_pair([0, 0], [0, 0])['dice']))

    def test_order_is_irrelevant(self):
        ref = table([1, 0, 1])
        got = calculate_sample_dice(align_labels(ref, ref.iloc[::-1]), 40)
        self.assertEqual(got.dice.iloc[0], 1)

    def test_missing_and_extra_ids_rejected(self):
        ref = table([1, 0, 1])
        for pred in [ref.iloc[:2], table([1, 0, 1], ['s0', 's1', 'extra'])]:
            with self.assertRaisesRegex(ValueError, 'Spot sets differ'):
                align_labels(ref, pred)

    def test_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            align_labels(table([1, 0]), table([1, 0], ['s0', 's0']))

    def test_nonbinary_and_missing_values_rejected(self):
        for values in [[0, 2], [0, np.nan], [0, 'False']]:
            with self.assertRaises(ValueError):
                align_labels(table([1, 0]), table(values))
        with self.assertRaises(ValueError):
            count_binary_pair([1], [1, 0])

    def test_macro_not_pooled(self):
        rows = pd.DataFrame([{'sample': 'S1', 'cutoff_percent': 40, 'dice': 1.0},
                             {'sample': 'S2', 'cutoff_percent': 40, 'dice': .2}])
        self.assertAlmostEqual(summarize(rows).mean_dice.iloc[0], .6)

    def test_input_schema_and_all_packaged_spots(self):
        root = Path(__file__).resolve().parent
        for filename in ['mask_hypo.csv'] + [f'hypo_{t}.csv' for t in CUTOFFS]:
            labels = read_labels(root / 'inputs' / filename)
            self.assertEqual(list(labels.columns), ['sample', 'spot_id', 'is_hypo'])
            self.assertEqual(len(labels), 22838)
            self.assertEqual(labels['sample'].nunique(), 8)

    def test_reproduce_packaged_results(self):
        root = Path(__file__).resolve().parent
        expected = pd.read_csv(root / 'results' / 'dice_per_sample.csv')
        with tempfile.TemporaryDirectory() as tmp:
            actual, _, _ = run(root / 'inputs', Path(tmp))
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
