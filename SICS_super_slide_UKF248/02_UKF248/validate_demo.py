"""Execute the SICS notebook and independently rerun a relocated copy."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import anndata as ad
from jupyter_client import KernelManager
from nbclient import NotebookClient
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / '01_super_slide/results/UKF_reference_raw.csv'
NOTEBOOK = 'UKF248_T_1_SICS_step_by_step.ipynb'
SCORE_COLUMNS = [
    'sample', 'pair', 'is_diagonal', 'observed_events', 'reference_mean_raw',
    'sample_total_raw', 'reference_total_raw', 'SICS_ratio', 'SICS_log2',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(folder, save=False):
    notebook = nbformat.read(folder / NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-X', 'utf8', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, timeout=180, kernel_name='python3', km=manager,
                            resources={'metadata': {'path': str(folder)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    assert not any(output.output_type == 'error' for cell in notebook.cells
                   if cell.cell_type == 'code' for output in cell.get('outputs', []))
    if save:
        nbformat.write(notebook, folder / NOTEBOOK.replace('.ipynb', '_executed.ipynb'))
    return sum(c.cell_type == 'code' for c in notebook.cells)


def main(write_checksums=True):
    before = {str(p): digest(p) for p in (ROOT / 'data').iterdir() if p.is_file()}
    before[str(REFERENCE)] = digest(REFERENCE)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_demo.py')
    with contextlib.redirect_stdout(stream):
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT / 'test_results.txt').write_text(stream.getvalue())
    assert tests.wasSuccessful()
    print('Tests passed:', tests.testsRun, flush=True)
    cells = execute(ROOT, save=True)
    print('Notebook executed:', cells, 'code cells', flush=True)
    all_pairs = pd.read_csv(ROOT / 'results/UKF248_T_1_pair_scores.csv', float_precision='round_trip')
    focal = pd.read_csv(ROOT / 'results/UKF248_T_1_Hypo_Core.csv', float_precision='round_trip')
    assert all_pairs.columns.tolist() == SCORE_COLUMNS
    assert focal.columns.tolist() == SCORE_COLUMNS
    assert len(focal) == 1 and len(all_pairs) == 21 and all_pairs.pair.is_unique
    assert all_pairs.is_diagonal.sum() == 6
    assert focal.iloc[0].observed_events == 803
    assert all_pairs.sample_total_raw.eq(38042).all()
    reference = pd.read_csv(REFERENCE, index_col=0, float_precision='round_trip')
    reference_total = float(np.tril(reference.to_numpy()).sum())
    np.testing.assert_allclose(all_pairs.reference_total_raw, reference_total)
    reference_metadata = json.loads(REFERENCE.with_name('run_manifest.json').read_text())
    assert digest(REFERENCE) == reference_metadata['reference_sha256']
    assert digest(REFERENCE) == digest(ROOT / 'results/UKF_reference_raw.csv')
    observed = pd.read_csv(ROOT / 'results/UKF248_T_1_observed_raw.csv', index_col=0)
    scores = pd.read_csv(ROOT / 'results/UKF248_T_1_score_matrix.csv', index_col=0, float_precision='round_trip')
    log2_scores = pd.read_csv(ROOT / 'results/UKF248_T_1_log2_score_matrix.csv', index_col=0,
                              float_precision='round_trip')
    for frame in [observed, scores, log2_scores]:
        pd.testing.assert_index_equal(frame.index, reference.index)
        pd.testing.assert_index_equal(frame.columns, reference.columns)
    expected_scores = ((observed + 1.0) / (reference + 1.0)
                       * np.tril(reference.to_numpy()).sum() / np.tril(observed.to_numpy()).sum())
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-12)
    np.testing.assert_allclose(focal.SICS_ratio, expected_scores.loc['MES_Hypo', 'MES_Core'], rtol=1e-12)
    assert np.isfinite(scores.to_numpy()).all() and (scores.to_numpy() > 0).all()
    assert np.isfinite(log2_scores.to_numpy()).all() and (log2_scores.to_numpy() < 0).any()
    np.testing.assert_allclose(log2_scores, np.log2(expected_scores), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(all_pairs.SICS_log2, np.log2(all_pairs.SICS_ratio), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(focal.SICS_log2, np.log2(focal.SICS_ratio), rtol=1e-12, atol=1e-14)
    for row in all_pairs.itertuples():
        a, b = row.pair.split('|')
        np.testing.assert_allclose(row.SICS_ratio, scores.loc[a, b], rtol=1e-12)
        np.testing.assert_allclose(row.SICS_log2, log2_scores.loc[a, b], rtol=1e-12, atol=1e-14)
        assert row.observed_events == observed.loc[a, b]
        np.testing.assert_allclose(row.reference_mean_raw, reference.loc[a, b], rtol=1e-12)
    focal_pair = all_pairs.loc[all_pairs.pair.eq('MES_Core|MES_Hypo')].reset_index(drop=True)
    pd.testing.assert_frame_equal(focal, focal_pair, check_exact=True)
    input_data = ad.read_h5ad(ROOT / 'data/UKF248_T_1_processed_spot_states.h5ad')
    result = ad.read_h5ad(ROOT / 'results/UKF248_T_1_demo_results.h5ad')
    assert len(input_data.layers) == 0 and len(result.layers) == 0
    assert input_data.raw is None and result.raw is None
    np.testing.assert_array_equal(input_data.X, result.X)
    assert input_data.obs_names.equals(result.obs_names)
    pd.testing.assert_frame_equal(input_data.obs, result.obs, check_exact=True)
    pd.testing.assert_frame_equal(input_data.var, result.var, check_exact=True)
    np.testing.assert_array_equal(input_data.obsm['spatial'], result.obsm['spatial'])
    assert (input_data.obsp['spatial_connectivities'] != result.obsp['spatial_connectivities']).nnz == 0
    assert set(result.uns) == set(input_data.uns) | {
        'sics_observed_raw', 'sics_reference_raw', 'sics_scores', 'sics_scores_log2',
        'sics_pair_scores', 'sics_total_mode',
    }
    assert result.uns['sics_pair_scores'].columns.tolist() == [
        column for column in SCORE_COLUMNS if column != 'pair'
    ]
    for key, frame in {
        'sics_observed_raw': observed, 'sics_reference_raw': reference, 'sics_scores': scores,
        'sics_scores_log2': log2_scores, 'sics_pair_scores': all_pairs.set_index('pair'),
    }.items():
        pd.testing.assert_frame_equal(result.uns[key], frame, check_exact=True)
    for key in input_data.uns:
        np.testing.assert_equal(input_data.uns[key], result.uns[key])
    manifest = json.loads((ROOT / 'results/run_manifest.json').read_text())
    assert manifest['score_scales'] == {
        'SICS_ratio': 'untransformed ratio', 'SICS_log2': 'log2(SICS_ratio)',
    }
    with tempfile.TemporaryDirectory(prefix='UKF248_relocated_') as temp:
        folder = Path(temp) / '02_UKF248'
        folder.mkdir()
        shutil.copy2(ROOT / NOTEBOOK, folder / NOTEBOOK)
        (folder / 'data').mkdir()
        for name in ['UKF248_T_1_processed_spot_states.h5ad']:
            shutil.copy2(ROOT / 'data' / name, folder / 'data' / name)
        upstream = folder.parent / '01_super_slide/results'
        upstream.mkdir(parents=True)
        for name in ['UKF_reference_raw.csv', 'run_manifest.json']:
            shutil.copy2(REFERENCE.parent / name, upstream / name)
        assert execute(folder) == cells
        repeat = pd.read_csv(folder / 'results/UKF248_T_1_pair_scores.csv', float_precision='round_trip')
        pd.testing.assert_frame_equal(all_pairs, repeat, check_exact=True)
        for path in (ROOT / 'results').glob('*.csv'):
            assert digest(path) == digest(folder / 'results' / path.name), path.name
        relocated = ad.read_h5ad(folder / 'results/UKF248_T_1_demo_results.h5ad')
        for key in ['sics_observed_raw', 'sics_reference_raw', 'sics_scores', 'sics_scores_log2', 'sics_pair_scores']:
            pd.testing.assert_frame_equal(result.uns[key], relocated.uns[key], check_exact=True)
        np.testing.assert_array_equal(result.X, relocated.X)
        pd.testing.assert_frame_equal(result.obs, relocated.obs, check_exact=True)
        pd.testing.assert_frame_equal(result.var, relocated.var, check_exact=True)
        np.testing.assert_array_equal(result.obsm['spatial'], relocated.obsm['spatial'])
        assert (result.obsp['spatial_connectivities'] != relocated.obsp['spatial_connectivities']).nnz == 0
    for path, expected in before.items():
        assert digest(path) == expected, path
    validation = {
        'tests_passed': tests.testsRun, 'notebook_code_cells_executed': cells,
        'relocated_notebook_uses_shared_super_slide_output': True,
        'relocated_results_match_exactly': True, 'inputs_unchanged': True,
        'sample': 'UKF248_T_1', 'spot_state_input_encoding': 'binary presence only',
        'no_continuous_layers_in_input_or_output': True,
        'labels_coordinates_and_graph_preserved': True,
        'n_spots': 3011, 'physical_edges': 8643, 'n_state_pairs': 21, 'focal_observed': 803,
        'sample_total_raw': 38042, 'reference_total_raw': reference_total,
        'reference_seed': reference_metadata['seed'], 'reference_rng': reference_metadata['rng'],
        'reference_sha256': digest(REFERENCE),
        'focal_SICS_ratio': float(focal.SICS_ratio.iloc[0]),
        'focal_SICS_log2': float(focal.SICS_log2.iloc[0]),
        'score_scales': manifest['score_scales'],
        'all_pair_log2_matches_ratio': True, 'matrix_and_anndata_scales_verified': True,
        'analysis_mode': 'state_pair_events, both totals use raw lower triangle',
    }
    (ROOT / 'verification.json').write_text(json.dumps(validation, indent=2) + '\n')
    if write_checksums:
        files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                       and p.name != 'SHA256SUMS')
        (ROOT / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
