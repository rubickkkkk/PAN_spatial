# Xenium PANZONER: ROI Demo

## Open

Open `demo.ipynb` for the executed workflow or `demo.html` for the browser view. The separate overview image is `assets/selected_region.png`; a high-resolution PNG, PDF, and SVG are also included.

## Input

`data/roi_input.h5ad` contains only the 1,798 cells in the selected ROI, including 381 Hypo cells. Cell IDs are in `adata.obs_names`, the binary label is in `adata.obs['is_hypo']`, and coordinates in micrometres are in `adata.obsm['spatial']`. The expression matrix has zero columns because the workflow uses only cell labels and coordinates. The overview is a static image and contains no additional computational input.

## Calculation

1. Load the ROI AnnData and plot its binary labels.
2. Build the 50 μm neighborhood graph.
3. Calculate Hypo neighbor fractions.
4. Classify Hypo cells.
5. Identify connected regions and retain regions with at least 50 cells.
6. Construct the PAN boundary and calculate signed distances.
7. Select Hypo cells within 30 μm inside and cells within 50 μm outside.
8. Plot the final zones and export the annotated ROI AnnData.

## Outputs

`results/roi_annotated.h5ad` contains the ROI cells, computed labels, distances, neighborhood graph, and boundary geometry. Each step has PNG, PDF, and SVG figures. Tabular outputs include `cell_annotations.tsv`, `gradient_cells.tsv`, and `zone_counts.tsv`.

## Run

Install `requirements.txt`, set `FONT` to an actual Arial font in Setup, and run `python execute_notebook.py`. Alternatively, open `demo_template.ipynb` from the demo directory and run all cells.
