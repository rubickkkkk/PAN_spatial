"""Validate binary labels, grid geometry, and state-pair counting."""
import ast
from pathlib import Path
import unittest

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent


class SuperSlideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / 'SICS_super_slide.ipynb', as_version=4)
        cls.scope = {'np': np, 'sparse': sparse, 'cKDTree': cKDTree}
        for cell in cls.notebook.cells:
            if cell.cell_type == 'code':
                nodes = [n for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef)]
                exec(compile(ast.Module(body=nodes, type_ignores=[]), '<functions>', 'exec'), cls.scope)
        cls.counter = staticmethod(cls.scope['count_state_pair_events'])
        cls.grid = staticmethod(cls.scope['build_hex_grid'])

    def test_input(self):
        data = ad.read_h5ad(ROOT / 'data/pooled_spot_states.h5ad')
        self.assertEqual(data.shape, (33872, 6))
        self.assertEqual(data.obs.shape[1], 0)
        self.assertEqual(data.obs_names.tolist(), [f'SPOT_{i:06d}' for i in range(1, 33873)])
        self.assertTrue(data.obs_names.is_unique)
        self.assertTrue(np.isin(data.X, [0, 1]).all())
        self.assertEqual(len(data.layers), 0)
        self.assertIsNone(data.raw)

    def test_presence_table(self):
        data = ad.read_h5ad(ROOT / 'data/pooled_spot_states.h5ad')
        table = pd.read_csv(ROOT / 'data/pooled_spot_state_presence.csv.gz', index_col=0)
        self.assertEqual(table.index.tolist(), data.obs_names.tolist())
        self.assertEqual(table.columns.tolist(), data.var_names.tolist())
        np.testing.assert_array_equal(table.to_numpy(), data.X)

    def test_presentation(self):
        markdown = [c.source for c in self.notebook.cells if c.cell_type == 'markdown']
        self.assertEqual(markdown, ['# SICS: Super-slide'])
        self.assertTrue(all(c.source.isascii() for c in self.notebook.cells))

    def test_diagonal_once(self):
        np.testing.assert_array_equal(self.counter([[1, 0], [1, 0]], [[0, 1]]), [[1, 0], [0, 0]])

    def test_multilabel_events(self):
        counts = self.counter([[1, 1], [1, 1]], [[0, 1]])
        np.testing.assert_array_equal(counts, [[1, 2], [2, 1]])
        self.assertEqual(np.tril(counts).sum(), 4)

    def test_grid(self):
        coords, edges, graph = self.grid(33872)
        self.assertEqual(len(np.unique(coords, axis=0)), 33872)
        self.assertEqual(len(edges), 100880)
        self.assertEqual((graph != graph.T).nnz, 0)
        self.assertEqual(graph.diagonal().sum(), 0)
        self.assertTrue((edges[:, 0] < edges[:, 1]).all())

    def test_seed_repeatability(self):
        np.random.seed(186)
        a = np.arange(49)
        recorded = []
        for _ in range(6):
            np.random.shuffle(a)
            recorded.append(a.copy())
        np.random.seed(186)
        b = np.arange(49)
        for expected in recorded:
            np.random.shuffle(b)
            np.testing.assert_array_equal(expected, b)
        source = '\n'.join(c.source for c in self.notebook.cells if c.cell_type == 'code')
        self.assertIn('SEED = 186', source)
        self.assertIn('np.random.seed(SEED)', source)
        self.assertEqual(source.count('np.random.shuffle(placement)'), 2)
        self.assertNotIn('default_rng', source)

    def test_coordinate_shuffle_equivalence(self):
        coordinates, _, _ = self.grid(49)
        np.random.seed(186)
        placement = np.arange(49)
        np.random.shuffle(placement)
        recorded = []
        for _ in range(50):
            np.random.shuffle(placement)
            recorded.append(placement.copy())
        np.random.seed(186)
        shuffled = [tuple(row) for row in coordinates]
        np.random.shuffle(shuffled)
        shuffled = np.asarray(shuffled)
        for expected in recorded:
            np.random.shuffle(shuffled)
            np.testing.assert_array_equal(shuffled, coordinates[expected])

    def test_sparse_count_equivalence(self):
        _, edges, graph = self.grid(49)
        np.random.seed(186)
        labels = np.random.randint(0, 2, size=(49, 6))
        counts = self.counter(labels, edges)
        expected = labels.T @ (graph @ labels)
        expected[np.diag_indices_from(expected)] //= 2
        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(np.tril(counts).sum(), (labels.T @ (graph @ labels)).sum() / 2)


if __name__ == '__main__':
    unittest.main()
