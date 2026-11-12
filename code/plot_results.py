"""Render saved results independently of MRI feature extraction or training."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path


def plot_confusion(out, matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=['HC', 'ULD']).plot(cmap='Blues')
    plt.title('BrainIAC frozen + Logistic Regression')
    plt.tight_layout()
    plt.savefig(Path(out) / 'confusion_matrix.png', dpi=180)
    plt.close()


def make_plots(out, y, probabilities, matrix):
    """Used by the full experiment and by independent saved-OOF plotting."""
    plot_confusion(out, matrix)
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y, probabilities)
    plt.title('ULD vs HC: pooled out-of-fold predictions')
    plt.tight_layout()
    plt.savefig(Path(out) / 'roc_curve.png', dpi=180)
    plt.close()


def main(argv=None):
    bundle = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=bundle.parent / 'results')
    p.add_argument('--output-dir', type=Path, default=bundle.parent / 'article/figures')
    args = p.parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.results_dir.resolve()
    metrics = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    oof = source / 'oof_predictions.csv'
    if oof.is_file():
        import pandas as pd
        table = pd.read_csv(oof)
        make_plots(out, table['label'], table['probability_ULD'], metrics['confusion_matrix'])
        roc_status = 'computed_from_saved_oof_predictions'
    else:
        plot_confusion(out, metrics['confusion_matrix'])
        original = source / 'original_notebook.ipynb'
        if original.is_file() and metrics.get('status') == 'original_notebook_saved_output':
            notebook = json.loads(original.read_text(encoding='utf-8'))
            images = [o['data']['image/png'] for cell in notebook['cells']
                for o in cell.get('outputs', []) if 'image/png' in o.get('data', {})]
            if len(images) != 2:
                raise ValueError('Expected the original confusion-matrix and ROC images.')
            (out / 'roc_curve.png').write_bytes(base64.b64decode(''.join(images[1])))
            roc_status = 'original_notebook_png_extracted_without_recomputing_curve'
        else:
            (out / 'roc_curve.png').unlink(missing_ok=True)
            roc_status = 'not_available_no_oof_predictions'
    provenance = {'source_results': str(source), 'roc_status': roc_status,
        'training_rerun': False, 'confusion_matrix_source': 'saved_metrics_json'}
    (out / 'plot_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Figures saved to', out)
    print('ROC:', roc_status)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
