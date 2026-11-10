# PAN spatial analysis demos

This repository provides three worked examples for spatial transcriptomics in glioblastoma. Each demo includes its inputs, Jupyter notebooks, results, and instructions for reproducing the analysis.

| Demo | Inputs and analysis | Start here |
| --- | --- | --- |
| PANZONER v3 | Identify PAN regions and six surrounding layers in a Visium section (UKF275_T_1), or compute distance zones in a Xenium region of interest. | [Visium](PANZONER_demos/Visium_PANZONER/README.md) / [Xenium](PANZONER_demos/Xenium_PANZONER/README.md) |
| Dice v4 | Compare five supplied PAN core predictions (30%, 35%, 40%, 45%, and 50% cutoffs) with the same independent manual reference annotations across eight samples. | [Instructions](Dice_binary_inputs_demo_20261108/README.md) / [Notebook](Dice_binary_inputs_demo_20261108/Dice_step_by_step.ipynb) |
| SICS v4 | Build a shared randomized super-slide reference and calculate state-pair adjacency scores for UKF248_T_1. The Python workflow exports both the normalized ratio and its log2 transformation. | [Instructions](SICS_super_slide_UKF248/README.md) / [1. Build the reference](SICS_super_slide_UKF248/01_super_slide/SICS_super_slide.ipynb) / [2. Score the sample](SICS_super_slide_UKF248/02_UKF248/UKF248_T_1_SICS_step_by_step.ipynb) |

To run a demo, clone or download the repository, install the dependencies listed in that demo's instructions, and open its notebook from the indicated directory. The PANZONER folders also include HTML copies of the executed notebooks.
