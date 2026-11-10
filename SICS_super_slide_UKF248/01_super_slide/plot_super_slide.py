"""Plot the exported super-slide grid and a local adjacency view."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FONT_ROOT = Path('/home/biotree/.local/share/fonts/msttcorefonts')


def main():
    for name in ['Arial.ttf', 'Arial_Bold.ttf']:
        path = FONT_ROOT / name
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    for weight in ['normal', 'bold']:
        prop = font_manager.FontProperties(family='Arial', weight=weight)
        path = font_manager.findfont(prop, fallback_to_default=False)
        if font_manager.FontProperties(fname=path).get_name() != 'Arial':
            raise RuntimeError('Actual Arial is required')
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
        'axes.titlesize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.linewidth': 0.55, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    coordinates_path = ROOT / 'results/grid_coordinates.csv'
    edges_path = ROOT / 'results/grid_edges.csv'
    xy = pd.read_csv(coordinates_path)[['x', 'y']].to_numpy()
    edges = pd.read_csv(edges_path)[['source', 'target']].to_numpy(dtype=np.int64)
    assert len(xy) == 33872 and len(edges) == 100880
    degree = np.bincount(edges.ravel(), minlength=len(xy))
    center = int(np.argmin(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)))
    assert degree[center] == 6
    cx, cy = xy[center]
    bounds = (cx - 425, cx + 425, cy - 365, cy + 365)
    in_view = ((xy[:, 0] >= bounds[0]) & (xy[:, 0] <= bounds[1])
               & (xy[:, 1] >= bounds[2]) & (xy[:, 1] <= bounds[3]))
    local_edges = edges[in_view[edges[:, 0]] & in_view[edges[:, 1]]]
    center_edges = edges[(edges == center).any(axis=1)]
    neighbors = np.unique(center_edges[center_edges != center])
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 91 / 25.4),
                             gridspec_kw={'width_ratios': [1.08, 1]}, layout='constrained')
    left, right = axes
    left.scatter(xy[:, 0], xy[:, 1], s=0.38, color='#49666B', linewidths=0)
    left.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2],
                             fill=False, edgecolor='#B83F3F', linewidth=0.8))
    left.set_xlim(xy[:, 0].min() - 450, xy[:, 0].max() + 450)
    left.set_ylim(xy[:, 1].max() + 450, xy[:, 1].min() - 450)
    left.set_xticks([12000, 18000, 24000, 30000])
    left.set_yticks([4000, 8000, 12000, 16000])
    left.set_title('Full grid | 33,872 spots', loc='left', pad=8)
    right.add_collection(LineCollection(xy[local_edges], colors='#CFD5D6', linewidths=0.45))
    right.scatter(xy[in_view, 0], xy[in_view, 1], s=12, color='#49666B', linewidths=0, zorder=3)
    right.add_collection(LineCollection(xy[center_edges], colors='#B83F3F', linewidths=1.0, zorder=4))
    right.scatter(xy[neighbors, 0], xy[neighbors, 1], s=24, color='#B83F3F', linewidths=0, zorder=5)
    right.scatter([cx], [cy], s=35, color='#B83F3F', edgecolors='white', linewidths=0.7, zorder=6)
    right.set_xlim(bounds[:2])
    right.set_ylim(bounds[3], bounds[2])
    right.set_xticks([cx - 300, cx, cx + 300])
    right.set_yticks([cy - 258, cy, cy + 258])
    right.set_title('Local view | six neighbors', loc='left', pad=8)
    for panel, ax in zip('AB', axes):
        ax.set_aspect('equal', adjustable='box', anchor='N')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(width=0.55, length=2.5)
        ax.set_xlabel('Grid X (coordinate units)')
        ax.set_ylabel('Grid Y (coordinate units)')
        ax.text(-0.13, 1.10, panel, transform=ax.transAxes, fontsize=10, fontweight='bold')
    fig.savefig(out / 'super_slide_spatial_layout.pdf', bbox_inches='tight')
    fig.savefig(out / 'super_slide_spatial_layout.svg', bbox_inches='tight')
    fig.savefig(out / 'super_slide_spatial_layout.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    local = pd.DataFrame(xy[in_view], columns=['x', 'y'])
    local.insert(0, 'grid_site_index', np.flatnonzero(in_view))
    local.to_csv(out / 'local_view_coordinates.csv', index=False)
    row_counts = pd.Series(xy[:, 1]).value_counts().sort_index()
    report = {
        'coordinate_file': 'results/grid_coordinates.csv',
        'coordinate_sha256': hashlib.sha256(coordinates_path.read_bytes()).hexdigest(),
        'n_spots': len(xy), 'n_edges': len(edges), 'grid_rows': len(row_counts),
        'spots_per_complete_row': int(row_counts.max()), 'spots_in_last_row': int(row_counts.iloc[-1]),
        'central_grid_site_index': center, 'central_node_degree': int(degree[center]),
        'coordinate_units': 'artificial grid coordinates; not micrometres',
        'pdf_geometry': 'vector', 'font': 'Arial', 'png_dpi': 600,
    }
    (out / 'spatial_layout_metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
