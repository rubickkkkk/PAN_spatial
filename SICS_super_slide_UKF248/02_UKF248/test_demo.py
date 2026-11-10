"""Check once-counted diagonals, lower-triangle totals, and UKF248 outputs."""
import ast
import contextlib
import io
import itertools
from pathlib import Path
import unittest

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / '01_super_slide/results/UKF_reference_raw.csv'


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adata = ad.read_h5ad(ROOT / 'data/UKF248_T_1_processed_spot_states.h5ad')
        cls.labels = np.asarray(cls.adata.X, dtype=np.int64)
        cls.graph = cls.adata.obsp['spatial_connectivities'].tocsr()
        upper = sparse.triu(cls.graph, k=1, format='coo')
        cls.edges = np.column_stack([upper.row, upper.col])
        cls.nb = nbformat.read(ROOT / 'UKF248_T_1_SICS_step_by_step.ipynb', as_version=4)
        nbformat.validate(cls.nb)
        scope = {'np': np}
        for cell in cls.nb.cells:
            if cell.cell_type == 'code':
                for node in ast.parse(cell.source).body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'count_state_pair_events':
                        exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook-counter>', 'exec'), scope)
        cls.counter = staticmethod(scope['count_state_pair_events'])
        cls.matrix = cls.counter(cls.labels, cls.edges)

    def test_notebook_presentation(self):
        markdown = [cell.source for cell in self.nb.cells if cell.cell_type == 'markdown']
        self.assertEqual(len(markdown), 1)
        self.assertTrue(markdown[0].startswith('# SICS: UKF248_T_1'))
        self.assertTrue(all(cell.source.isascii() for cell in self.nb.cells))
        self.assertEqual(sum(cell.cell_type == 'code' for cell in self.nb.cells), 9)

    def test_input_labels(self):
        self.assertEqual(self.adata.shape, (3011, 6))
        self.assertTrue(np.isin(self.labels, [0, 1]).all())
        self.assertEqual(len(self.adata.layers), 0)
        self.assertIsNone(self.adata.raw)
        self.assertEqual(self.labels[:, 2].sum(), 953)
        self.assertEqual(self.labels[:, 5].sum(), 526)
        self.assertEqual((self.labels[:, 2] * self.labels[:, 5]).sum(), 124)

    def test_presence_csv(self):
        presence = pd.read_csv(ROOT / 'data/UKF248_T_1_spot_state_presence.csv', index_col=0)
        self.assertEqual(presence.index.tolist(), self.adata.obs_names.tolist())
        self.assertEqual(presence.columns.tolist(), self.adata.var_names.tolist())
        np.testing.assert_array_equal(presence.to_numpy(), self.labels)
        np.testing.assert_array_equal(self.adata.obs.n_positive_states.to_numpy(), self.labels.sum(axis=1))
        for j, state in enumerate(self.adata.var_names):
            np.testing.assert_array_equal(self.adata.obs[f'{state}_positive'].to_numpy(), self.labels[:, j].astype(bool))

    def test_mixture_strings(self):
        for value, row in zip(self.adata.obs.tumor_state_mixture, self.labels):
            self.assertEqual(value, '|'.join(self.adata.var_names[row.astype(bool)]) or 'Unassigned')

    def test_graph(self):
        self.assertEqual(len(self.edges), 8643)
        self.assertEqual((self.graph != self.graph.T).nnz, 0)
        self.assertEqual(self.graph.diagonal().sum(), 0)

    def test_diagonal_once(self):
        np.testing.assert_array_equal(self.counter([[1, 0], [1, 0]], [[0, 1]]), [[1, 0], [0, 0]])

    def test_multilabel_edge(self):
        matrix = self.counter([[1, 1], [1, 1]], [[0, 1]])
        np.testing.assert_array_equal(matrix, [[1, 2], [2, 1]])
        self.assertEqual(np.tril(matrix).sum(), 4)

    def test_invalid_edges(self):
        for edges in [[[0, 0]], [[1, 0]], [[0, 2]], [[0, 1], [0, 1]], [[0, 0.5]]]:
            with self.assertRaises(ValueError):
                self.counter([[1, 0], [0, 1]], edges)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            self.counter([[0.1, 0.9], [1, 0]], [[0, 1]])

    def test_empty_edges(self):
        self.assertEqual(self.counter([[1, 0]], np.empty((0, 2), dtype=int)).sum(), 0)

    def test_real_count_and_total(self):
        self.assertEqual(self.matrix[2, 5], 803)
        self.assertEqual(np.tril(self.matrix).sum(), 38042)
        directed = self.labels.T @ (self.graph @ self.labels)
        directed[np.diag_indices_from(directed)] //= 2
        np.testing.assert_array_equal(self.matrix, directed)

    def test_repeated_values_are_not_deduplicated(self):
        self.assertNotEqual(np.unique(self.matrix).sum(), np.tril(self.matrix).sum())
        self.assertEqual(self.matrix[2, 2], self.matrix[3, 5])

    def test_reference_matrix(self):
        frame = pd.read_csv(REFERENCE, index_col=0)
        self.assertEqual(frame.index.tolist(), self.adata.var_names.tolist())
        self.assertEqual(frame.columns.tolist(), self.adata.var_names.tolist())
        reference = frame.to_numpy()
        self.assertTrue(np.isfinite(reference).all() and (reference >= 0).all())
        np.testing.assert_allclose(reference, reference.T, rtol=0, atol=1e-9)
        with np.load(REFERENCE.with_name('super_slide_draws.npz')) as z:
            np.testing.assert_allclose(reference, z['draws'].mean(axis=0), rtol=0, atol=1e-9)
        self.assertGreater(np.tril(reference).sum(), 0)

    def test_total_counts_all_state_combinations(self):
        for labels in [np.array([[1, 1], [1, 0]]), np.array([[0, 1], [1, 1]])]:
            matrix = self.counter(labels, [[0, 1]])
            self.assertEqual(np.tril(matrix).sum(), labels[0].sum() * labels[1].sum())

    def test_sics_score(self):
        ref = pd.read_csv(REFERENCE, index_col=0)
        ref = ref.loc[self.adata.var_names, self.adata.var_names].to_numpy()
        score = 804 / (ref[2, 5] + 1) * np.tril(ref).sum() / np.tril(self.matrix).sum()
        self.assertTrue(np.isfinite(score) and score > 0)
        source = '\n'.join(c.source for c in self.nb.cells if c.cell_type == 'code')
        self.assertIn("BASE.parent / '01_super_slide/results/UKF_reference_raw.csv'", source)
        self.assertIn('(matrix + 1.0) / (reference.to_numpy() + 1.0)', source)

    def test_counting_formula(self):
        code = '\n'.join(c.source for c in self.nb.cells if c.cell_type == 'code')
        self.assertIn('if idx1 == idx2:', code)
        self.assertIn('connect_sum = np.tril(matrix, k=0).sum()', code)
        self.assertNotIn('np.unique(matrix).sum()', code)
        self.assertIn('reference_connect_sum / sample_connect_sum', code)

    def score_scope(self, matrix=None, reference=None, totals=None):
        states = self.adata.var_names.tolist()
        matrix = self.matrix if matrix is None else matrix
        if reference is None:
            reference = pd.read_csv(REFERENCE, index_col=0, float_precision='round_trip').loc[states, states]
        reference_total, sample_total = (np.tril(reference).sum(), np.tril(matrix).sum()) if totals is None else totals
        scope = {
            'np': np, 'pd': pd, 'itertools': itertools, 'matrix': matrix,
            'reference': reference, 'states': states, 'PAIR': ('MES_Hypo', 'MES_Core'),
            'SAMPLE': 'UKF248_T_1', 'reference_connect_sum': reference_total,
            'sample_connect_sum': sample_total, 'observed_hypo_core': matrix[2, 5],
            'expected_hypo_core': reference.iloc[2, 5], 'display': lambda *values: None,
        }
        cell = next(c for c in self.nb.cells if c.cell_type == 'code' and 'score_values =' in c.source)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, '<notebook-score>', 'exec'), scope)
        return scope

    def test_log2_ratio_scale(self):
        matrix = np.full((6, 6), 3.0)
        np.fill_diagonal(matrix, [0, 1, 3, 7, 15, 3])
        states = self.adata.var_names.tolist()
        reference = pd.DataFrame(np.full((6, 6), 3.0), index=states, columns=states)
        scope = self.score_scope(matrix, reference, totals=(1.0, 1.0))
        np.testing.assert_array_equal(np.diag(scope['score_values'])[:5], [0.25, 0.5, 1, 2, 4])
        self.assertIn('log2_score_values', scope.keys())
        np.testing.assert_array_equal(np.diag(scope['log2_score_values'])[:5], [-2, -1, 0, 1, 2])

    def test_log2_requires_positive_finite_ratios(self):
        states = self.adata.var_names.tolist()
        reference = pd.DataFrame(np.full((6, 6), 3.0), index=states, columns=states)
        for value in [-1.0, -2.0, np.nan, np.inf]:
            with self.subTest(matrix_value=value):
                matrix = np.full((6, 6), 3.0)
                matrix[0, 0] = value
                with self.assertRaises(ValueError):
                    self.score_scope(matrix, reference, totals=(1.0, 1.0))

    def test_real_sample_log2_and_ratio(self):
        scope = self.score_scope()
        ratios = pd.read_csv(ROOT / 'results/UKF248_T_1_score_matrix.csv', index_col=0,
                             float_precision='round_trip')
        np.testing.assert_allclose(scope['score_values'], ratios, rtol=1e-14, atol=0)
        self.assertAlmostEqual(scope['focal_ratio'], 1.6283903321491158, places=14)
        self.assertIn('log2_score_values', scope.keys())
        np.testing.assert_allclose(scope['log2_score_values'], np.log2(ratios), rtol=1e-14, atol=0)
        self.assertAlmostEqual(scope['focal_log2'], np.log2(1.6283903321491158), places=14)
        self.assertTrue((scope['log2_score_values'] < 0).any())

    def test_log2_pair_table(self):
        scope = self.score_scope()
        cell = next(c for c in self.nb.cells if c.cell_type == 'code' and 'records = []' in c.source)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, '<notebook-pairs>', 'exec'), scope)
        pairs = scope['pair_scores']
        self.assertEqual(len(pairs), 21)
        self.assertIn('SICS_log2', pairs.columns)
        np.testing.assert_allclose(pairs.SICS_log2, np.log2(pairs.SICS_ratio), rtol=1e-14, atol=0)
        self.assertEqual(scope['focal'].SICS_log2.iloc[0], scope['focal_log2'])


if __name__ == '__main__':
    unittest.main()
