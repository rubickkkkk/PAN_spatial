# Dice From Binary Spot Labels

`inputs/mask_hypo.csv` contains the independent manual annotations used as the reference for Dice comparison.
All five threshold prediction tables are evaluated against this same fixed mask.

## Inputs

All six CSV files contain exactly `sample`, `spot_id`, and `is_hypo`.
The only measurement field is `is_hypo`: 1 means positive, 0 means negative.
There are no abundances, fractions, expression values, coordinates, or images.
Every file contains the same 22,838 spots from eight samples, including all
negative spots. Sample and spot IDs are retained solely for alignment.

- `inputs/mask_hypo.csv`: the fixed reference mask.
- `inputs/hypo_30.csv`: PAN region labels at the 30% cutoff.
- `inputs/hypo_35.csv`: PAN region labels at the 35% cutoff.
- `inputs/hypo_40.csv`: PAN region labels at the 40% cutoff.
- `inputs/hypo_45.csv`: PAN region labels at the 45% cutoff.
- `inputs/hypo_50.csv`: PAN region labels at the 50% cutoff.

The threshold labels use the processed `PAN_core == True`, equivalent to
`PAN_label == 'Core'`.
They are not a fresh application of a numeric cutoff to raw spot fractions,
and they are not `cell_status == 'core'`. The labels are already supplied;
the demo does not reconstruct thresholds or spatial regions.

## Calculation

For each sample and cutoff, align the two binary tables using the complete
`(sample, spot_id)` keys. Missing, additional, duplicate or non-binary entries
raise an error. No inner join, row filtering, or position-based alignment is used.

`TP = number of spots positive in both tables`

`FP = number positive only in the threshold prediction`

`FN = number positive only in the reference mask`

`Dice = 2 * TP / (N_mask_positive + N_prediction_positive)`

`Dice = 2 * TP / (2 * TP + FP + FN)`

TN is recorded for completeness but does not enter Dice. If both positive sets
are empty, Dice is undefined (NaN), not automatically set to 1. This does not
occur in these inputs. The primary summary is the arithmetic mean of the
eight sample Dice values, not a pooled-spot Dice. Each sample uses exactly
the same reference mask for every cutoff.

## Run

Open `Dice_step_by_step.ipynb` and run its cells in order. The completed copy is
`Dice_step_by_step.executed.ipynb`. No source analysis directory is needed;
keep the notebook, `calculate_dice.py`, and `inputs/` together.

```bash
python calculate_dice.py
python -m unittest test_calculate_dice -v
```

The CLI writes `results/`; the notebook writes `results_notebook/`.
Outputs include per-sample counts and Dice, the equally weighted means, and
a sample-by-cutoff table. `provenance.json` records the input derivation and
checksums. `verification.json` records reproduction and relocation checks.

`requirements.txt` pins the versions used for this execution. NumPy and pandas
are sufficient for the CLI; Jupyter is needed only for the notebook.
