# Visium PANZONER Demo

## Open

Open `demo.ipynb` for the executed workflow or `demo.html` for the browser view.

## Input

`data/UKF275_T_1_binary_input.tsv` contains spot IDs, coordinates, and binary Hypo labels. The H&E background is included in `data/`. The overlay uses the palette, legend labels, and crop in `data/plot_style.json`.

## Calculation

Build the spatial graph, classify Hypo spots, identify and expand connected regions, merge neighboring regions, retain regions with at least 20 spots, and generate six outward layers. The two growth passes update labels in input row order, so changing that order can change region membership. Keep the supplied spot order to reproduce this example.

## Outputs

The final H&E overlay is saved as `results/figures/pan_region_and_layers` in PNG, PDF, and SVG formats. Intermediate spot labels, counts, the spatial graph, and final annotations are in `results/`.

## Run

Install `requirements.txt`, set the font path to an actual Arial font, and run `python execute_notebook.py` from this directory. Alternatively, open `demo_template.ipynb` and run all cells.
