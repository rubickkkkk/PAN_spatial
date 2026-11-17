"""Build a portable notebook from the supplied SICS inputs."""
from pathlib import Path
import platform
from textwrap import dedent

import anndata as ad
import nbformat as nbf
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
SAMPLE = 'UKF248_T_1'
NOTEBOOK = 'UKF248_T_1_SICS_step_by_step.ipynb'


def make_notebook():
    cells = [nbf.v4.new_markdown_cell(dedent('''
    # SICS: UKF248_T_1

    This notebook reports the normalized ratio (`SICS_ratio`) and its base-2 logarithm
    (`SICS_log2 = log2(SICS_ratio)`), computed in Python. Negative, zero, and positive
    log2 scores correspond to ratios below, equal to, and above 1, respectively.
    The +1 pseudocount is included in the ratio formula; log2 uses that ratio directly.
    ''').strip())]

    def code(source):
        cells.append(nbf.v4.new_code_cell(dedent(source).strip()))

    code('''
    from pathlib import Path
    import hashlib
    import json
    import itertools
    import shutil
    from importlib.metadata import version

    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from IPython.display import display

    BASE = Path.cwd()
    INPUT = BASE / 'data/UKF248_T_1_processed_spot_states.h5ad'
    REFERENCE = BASE.parent / '01_super_slide/results/UKF_reference_raw.csv'
    REFERENCE_MANIFEST = REFERENCE.with_name('run_manifest.json')
    OUT = BASE / 'results'
    if not INPUT.is_file():
        raise FileNotFoundError('Start the kernel in the demo folder, or set BASE to that folder.')
    if not REFERENCE.is_file():
        raise FileNotFoundError('Run 01_super_slide/SICS_super_slide.ipynb first.')
    OUT.mkdir(exist_ok=True)
    SAMPLE = 'UKF248_T_1'
    PAIR = ('MES_Hypo', 'MES_Core')
    print({p: version(p) for p in ['anndata', 'numpy', 'pandas', 'scipy']})
    ''')
    code('''
    adata = ad.read_h5ad(INPUT)
    states = adata.var_names.tolist()
    assert states == ['AC_like', 'G_cycling', 'MES_Hypo', 'NPC_like', 'OPC_like', 'MES_Core']
    assert adata.shape == (3011, 6)
    assert adata.obs_names.is_unique and adata.var_names.is_unique
    assert set(adata.obs['sample'].astype(str)) == {SAMPLE}
    labels = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    assert np.isin(labels, [0, 1]).all()
    labels = labels.astype(np.int64)
    assert len(adata.layers) == 0 and adata.raw is None
    expected_mixtures = ['|'.join(name for name, present in zip(states, row) if present)
                         or 'Unassigned' for row in labels]
    assert adata.obs['tumor_state_mixture'].astype(str).tolist() == expected_mixtures
    np.testing.assert_array_equal(adata.obs['n_positive_states'].to_numpy(), labels.sum(axis=1))
    for j, state in enumerate(states):
        np.testing.assert_array_equal(adata.obs[f'{state}_positive'].to_numpy(), labels[:, j].astype(bool))
    print(adata)
    display(adata.obs[['sample', 'tumor_state_mixture', 'n_positive_states']].head(10))
    display(pd.DataFrame(labels, index=adata.obs_names, columns=states).head(10))
    display(pd.Series(labels.sum(axis=0), index=states, name='positive_spots').to_frame())
    ''')
    code('''
    graph = adata.obsp['spatial_connectivities'].tocsr(copy=True)
    graph.eliminate_zeros()
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert not graph.diagonal().any() and (graph != graph.T).nnz == 0
    assert np.isin(graph.data, [1]).all()
    upper = sparse.triu(graph, k=1, format='coo')
    edges = np.column_stack([upper.row, upper.col]).astype(np.int64)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    assert len(edges) == 8643
    assert (edges[:, 0] < edges[:, 1]).all()
    assert len(np.unique(edges, axis=0)) == len(edges)
    edge_table = pd.DataFrame({
        'source': edges[:, 0], 'target': edges[:, 1],
        'source_spot': adata.obs_names[edges[:, 0]],
        'target_spot': adata.obs_names[edges[:, 1]],
    })
    print(f'{adata.n_obs} spots; {len(edges)} undirected physical edges')
    display(edge_table.head(10))
    ''')
    code('''
    def count_state_pair_events(labels, edges):
        labels = np.asarray(labels)
        if labels.ndim != 2 or not np.isin(labels, [0, 1]).all():
            raise ValueError('Expected a binary spot-by-state matrix')
        edges = np.asarray(edges)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('Expected two indices per edge')
        if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
            raise ValueError('Edge indices must be finite integers')
        edges = edges.astype(np.int64)
        if edges.size and (edges.min() < 0 or edges.max() >= len(labels)):
            raise ValueError('Edge index outside label table')
        if np.any(edges[:, 0] >= edges[:, 1]) or len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError('Supply each edge once, with source < target')
        present = [np.flatnonzero(row) for row in labels]
        matrix = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.int64)
        for source, target in edges:
            for idx1 in present[source]:
                for idx2 in present[target]:
                    matrix[idx1, idx2] += 1
                    if idx1 == idx2:
                        continue
                    matrix[idx2, idx1] += 1
        return matrix

    matrix = count_state_pair_events(labels, edges)
    observed = pd.DataFrame(matrix, index=states, columns=states)
    i_hypo, i_core = [states.index(state) for state in PAIR]
    observed_hypo_core = int(matrix[i_hypo, i_core])
    assert observed_hypo_core == 803
    display(observed)
    print('Observed MES_Hypo--MES_Core events:', observed_hypo_core)
    ''')
    code('''
    connect_sum = np.tril(matrix, k=0).sum()
    sample_connect_sum = float(connect_sum)
    assert sample_connect_sum == 38042
    directed = labels.T @ (graph @ labels)
    expected = directed.copy()
    expected[np.diag_indices_from(expected)] //= 2
    np.testing.assert_array_equal(matrix, expected)
    assert sample_connect_sum == directed.sum() / 2
    print('Sample raw-event total:', sample_connect_sum)
    ''')
    code('''
    def align_reference(path, states):
        frame = pd.read_csv(path, index_col=0, float_precision='round_trip')
        assert frame.index.is_unique and frame.columns.is_unique
        assert set(frame.index) == set(states) and set(frame.columns) == set(states)
        frame = frame.loc[states, states]
        values = frame.to_numpy(dtype=float)
        assert np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values, values.T, rtol=0, atol=1e-10)
        return frame

    reference = align_reference(REFERENCE, states)
    reference_connect_sum = float(np.tril(reference.to_numpy(), k=0).sum())
    reference_metadata = json.loads(REFERENCE_MANIFEST.read_text())
    assert reference_metadata['reference_sha256'] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    assert np.isclose(reference_connect_sum, reference_metadata['connect_sum_raw'], rtol=0, atol=1e-8)
    expected_hypo_core = float(reference.loc[PAIR[0], PAIR[1]])
    assert reference_connect_sum > 0
    display(reference)
    print('Reference raw-event total:', reference_connect_sum)
    ''')
    code('''
    score_values = ((matrix + 1.0) / (reference.to_numpy() + 1.0)
                    * reference_connect_sum / sample_connect_sum)
    if not np.isfinite(score_values).all() or not (score_values > 0).all():
        raise ValueError('SICS ratios must be finite and positive before log2 transformation')
    log2_score_values = np.log2(score_values)
    assert np.isfinite(log2_score_values).all()
    score_matrix = pd.DataFrame(score_values, index=states, columns=states)
    log2_score_matrix = pd.DataFrame(log2_score_values, index=states, columns=states)
    focal_ratio = float(score_matrix.loc[PAIR[0], PAIR[1]])
    focal_log2 = float(log2_score_matrix.loc[PAIR[0], PAIR[1]])
    print(f'({observed_hypo_core} + 1) / ({expected_hypo_core:.2f} + 1) '
          f'* {reference_connect_sum:.2f} / {sample_connect_sum:.0f}')
    print(f'MES_Hypo--MES_Core ratio = {focal_ratio:.12f}')
    print(f'MES_Hypo--MES_Core log2 score = {focal_log2:.12f}')
    print('SICS ratio matrix:')
    display(score_matrix)
    print('SICS log2 matrix:')
    display(log2_score_matrix)
    ''')
    code('''
    records = []
    for i, j in itertools.combinations_with_replacement(range(len(states)), 2):
        pair = '|'.join(sorted((states[i], states[j])))
        value = float(score_values[i, j])
        records.append({
            'sample': SAMPLE, 'pair': pair, 'is_diagonal': i == j,
            'observed_events': int(matrix[i, j]), 'reference_mean_raw': float(reference.iloc[i, j]),
            'sample_total_raw': sample_connect_sum, 'reference_total_raw': reference_connect_sum,
            'SICS_ratio': value, 'SICS_log2': float(log2_score_values[i, j]),
        })
    pair_scores = pd.DataFrame(records)
    assert len(pair_scores) == 21 and pair_scores['pair'].is_unique
    focal = pair_scores.loc[pair_scores['pair'].eq('|'.join(sorted(PAIR)))].copy()
    display(focal)
    display(pair_scores)
    ''')
    code('''
    observed.to_csv(OUT / 'UKF248_T_1_observed_raw.csv', lineterminator='\\n')
    shutil.copyfile(REFERENCE, OUT / 'UKF_reference_raw.csv')
    score_matrix.to_csv(OUT / 'UKF248_T_1_score_matrix.csv', lineterminator='\\n')
    log2_score_matrix.to_csv(OUT / 'UKF248_T_1_log2_score_matrix.csv', lineterminator='\\n')
    pair_scores.to_csv(OUT / 'UKF248_T_1_pair_scores.csv', index=False, lineterminator='\\n')
    focal.to_csv(OUT / 'UKF248_T_1_Hypo_Core.csv', index=False, lineterminator='\\n')
    edge_table.to_csv(OUT / 'UKF248_T_1_undirected_edges.csv', index=False, lineterminator='\\n')
    manifest = {
        'sample': SAMPLE, 'n_spots': adata.n_obs, 'physical_edges': len(edges),
        'counting': 'same-state once via if-guard, off-diagonal symmetric',
        'total': 'lower triangle including diagonal, no pseudocounts on either side',
        'formula': '(C + 1)/(E_raw + 1) * reference_total_raw/sample_total_raw',
        'sample_total_raw': sample_connect_sum, 'reference_total_raw': reference_connect_sum,
        'spot_state_input': 'precomputed 0/1 presence labels; no continuous state layers',
        'scope': 'single-sample demonstration; no P values or group reassignment',
        'score_scales': {'SICS_ratio': 'untransformed ratio', 'SICS_log2': 'log2(SICS_ratio)'},
        'reference_seed': reference_metadata['seed'], 'reference_rng': reference_metadata['rng'],
        'reference_random_api': reference_metadata['random_api'],
        'sha256': {p.relative_to(BASE.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [INPUT, REFERENCE]},
        'versions': {p: version(p) for p in ['anndata', 'numpy', 'pandas', 'scipy']},
    }
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
    result = adata.copy()
    result.uns['sics_observed_raw'] = observed
    result.uns['sics_reference_raw'] = reference
    result.uns['sics_scores'] = score_matrix
    result.uns['sics_scores_log2'] = log2_score_matrix
    result.uns['sics_pair_scores'] = pair_scores.set_index('pair')
    result.uns['sics_total_mode'] = 'both raw lower-triangle totals, no pseudocounts'
    result.write_h5ad(OUT / 'UKF248_T_1_demo_results.h5ad', compression='gzip')
    print('Saved single-sample results:', OUT.relative_to(BASE))
    ''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata.language_info = {'name': 'python', 'version': platform.python_version()}
    return nb


def main():
    data = ROOT / 'data'
    adata = ad.read_h5ad(data / f'{SAMPLE}_processed_spot_states.h5ad')
    assert adata.shape == (3011, 6)
    assert len(adata.layers) == 0 and adata.raw is None
    labels = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    assert np.isin(labels, [0, 1]).all()
    presence = pd.read_csv(data / f'{SAMPLE}_spot_state_presence.csv', index_col=0)
    assert presence.index.tolist() == adata.obs_names.tolist()
    assert presence.columns.tolist() == adata.var_names.tolist()
    np.testing.assert_array_equal(presence.to_numpy(), labels)
    reference = pd.read_csv(ROOT.parent / '01_super_slide/results/UKF_reference_raw.csv', index_col=0,
                            float_precision='round_trip').loc[adata.var_names, adata.var_names]
    np.testing.assert_allclose(reference.to_numpy(), reference.to_numpy().T, rtol=0, atol=1e-10)
    assert np.isfinite(reference.to_numpy()).all() and np.tril(reference.to_numpy()).sum() > 0
    nb = make_notebook()
    nbf.validate(nb)
    nbf.write(nb, ROOT / NOTEBOOK)
    print('Notebook:', ROOT / NOTEBOOK)


if __name__ == '__main__':
    main()
