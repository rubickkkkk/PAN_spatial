PANZoner: Spatial Zonation of Pseudopalisading Necrosis

PANZoner is a spatial analysis workflow for identifying and annotating hypoxia-associated pseudopalisading necrosis (PAN) regions in glioblastoma spatial transcriptomics data.

--------------------------------------------
Input

AnnData (.h5ad) with gene expression, spatial coordinates, and hypoxia-related scores.

Method (brief)

Identify hypoxic spots based on hypoxia gene expression.

Construct spatial neighborhood graphs.

Classify hypoxic spots into core, border, and peripheral layers.

Segment and merge spatially connected PAN regions.

Output

Annotated AnnData object with PAN region and layer labels.

Spot-level annotation tables and spatial visualization figures.

Dependencies
Python ≥3.8, scanpy, squidpy, anndata, numpy, pandas, scipy, matplotlib.

--------------------------------------------------------------
SICS: Spatial Identity and Continuity Scoring

SICS is a framework for quantifying continuous spatial transitions of transcriptional states in glioblastoma tissue using spatial transcriptomics data.

---------------------------------------
Input

AnnData (.h5ad) with spatial coordinates and cell-state or program scores.

Method (brief)

Project cell-state scores onto spatial coordinates.

Quantify spatial continuity and gradual state transitions across tissue.

Identify dominant spatial transition patterns.

Output

Spatial continuity scores and state gradients stored in AnnData.

Publication-ready spatial plots.

Usage

Implemented in SICS_workflow.ipynb.

Dependencies
Python ≥3.8, scanpy, squidpy, numpy, pandas, matplotlib.
