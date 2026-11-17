# SICS: UKF248_T_1

Run `../01_super_slide/SICS_super_slide.ipynb` first with seed 186.
Open `UKF248_T_1_SICS_step_by_step.ipynb` using a Python kernel with the listed dependencies.
Run the cells from top to bottom. The executed copy ends in `_executed.ipynb`.

The notebook contains one introductory Markdown cell and nine code cells: input loading, binary
state labels, spatial edges, state-pair counting, the sample total, the reference
total, SICS ratios and log2 scores, a state-pair result table, and output export.

Each of the 3,011 spots has six binary state labels: 1 means annotated present,
and 0 means not annotated present. A spot may have multiple labels. No continuous
state layers are stored.

```python
matrix[idx1, idx2] += 1
if idx1 == idx2:
    continue
matrix[idx2, idx1] += 1

connect_sum = np.tril(matrix, k=0).sum()
SICS_ratio = (C + 1) / (E + 1) * reference_connect_sum / sample_connect_sum
SICS_log2 = np.log2(SICS_ratio)
```

Same-state events are counted once; different-state entries are symmetric.
Both totals include the diagonal and exclude pseudocounts. `C` is the observed
state-pair count and `E` is its mean count in the shared reference. The
pseudocount of 1 is added to these pair-specific counts before normalization.

The final score is computed directly in Python by applying `np.log2` once to
the finite, strictly positive normalized ratio. No additional pseudocount or
clipping is applied at this step. Ratios below 1 produce negative scores, a
ratio of 1 produces 0, and ratios above 1 produce positive scores. Both scales
are retained. P values and HiLink/LoLink assignments are outside this demo.

```text
Spots = 3011
Undirected spatial edges = 8643
MES_Hypo--MES_Core events = 803
Sample raw-event total = 38042
MES_Hypo--MES_Core SICS_ratio = 1.6283903321491158
MES_Hypo--MES_Core SICS_log2 = 0.7034465612491638
```

The reference is read directly from `../01_super_slide/results/UKF_reference_raw.csv`.
The reference total, MES_Hypo--MES_Core mean, SICS ratio, and log2 score are displayed during
execution and saved in `results/UKF248_T_1_Hypo_Core.csv`.
Keep both module directories together. Outputs are written to `results/`.
The notebook does not require Squidpy. Input spot labels, coordinates, and
spatial edges remain unchanged. A single-sample demonstration is not a
cohort-wide robustness analysis.

Outputs in `results/` include:

- `UKF248_T_1_pair_scores.csv`: all 21 unordered state pairs, with `SICS_ratio`
  and `SICS_log2` columns.
- `UKF248_T_1_Hypo_Core.csv`: the MES_Hypo--MES_Core row, with both score scales.
- `UKF248_T_1_score_matrix.csv`: the 6-by-6 ratio matrix.
- `UKF248_T_1_log2_score_matrix.csv`: the 6-by-6 log2 score matrix.
- `UKF248_T_1_demo_results.h5ad`: input labels and graph plus result tables.
  `uns['sics_scores']` stores ratios, `uns['sics_scores_log2']` stores log2
  scores, and `uns['sics_pair_scores']` includes both columns.

Run the tests, or execute and validate the notebook and its relocated copy:

```bash
python -m unittest -v test_demo
python validate_demo.py
```
