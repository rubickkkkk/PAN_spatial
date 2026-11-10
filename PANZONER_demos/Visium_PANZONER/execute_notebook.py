"""Execute with this Python environment; save a notebook with outputs and HTML."""
from pathlib import Path
import os
import sys
import json
import tempfile
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')

with tempfile.TemporaryDirectory(prefix='panzoner_kernel_') as temporary:
    spec = Path(temporary) / 'panzoner-local'
    spec.mkdir()
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'PANZONER local Python', 'language': 'python'}))
    km = KernelManager(kernel_name='panzoner-local',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False))
    nb = nbformat.read(ROOT / 'demo_template.ipynb', as_version=4)
    client = NotebookClient(nb, km=km, timeout=180,
        resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbformat.write(nb, ROOT / 'demo.ipynb')

exporter = HTMLExporter()
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
html, _ = exporter.from_notebook_node(nb)
(ROOT / 'demo.html').write_text(html)
print('Saved', ROOT / 'demo.ipynb')
print('Saved', ROOT / 'demo.html')
