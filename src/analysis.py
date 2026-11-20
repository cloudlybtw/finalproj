"""Deterministic analysis of frozen Fashion-MNIST test predictions.

No model fitting or selection occurs here. Seed 11 was designated for illustrative
matrices/images before test evaluation; aggregate scores always use every seed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import LABELS, read_idx, save_json
from src.study import verify_saved

COLORS = {'logistic': '#52657A', 'cnn': '#147D92', 'dropout': '#CB6C32'}
SEED_COLORS = ['#147D92', '#CB6C32', '#775DA6']
DISPLAY = {'logistic': 'Logistic regression', 'cnn': 'CNN', 'dropout': 'CNN + dropout'}
SHORT_LABELS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
REPRESENTATIVE_SEED = 11


def _save(fig, folder, name):
    """Keep vector and raster copies of each figure."""
    for extension in ('pdf', 'png'):
        fig.savefig(folder / f'{name}.{extension}', dpi=220,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#B8C2CC',
        'text.color': '#182B3A', 'axes.labelcolor': '#182B3A',
        'xtick.color': '#435363', 'ytick.color': '#435363',
        'grid.color': '#E1E7EB', 'grid.linewidth': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def build_analysis(root=Path('.')):
    """Write auditable summary tables and figures from all seven saved runs.

    Returns file paths and prediction-verification status. CSV scores remain in
    [0, 1]; figures display percentages. Standard deviations use ddof=1 and are
    undefined (blank CSV fields) for the single logistic-regression fit.
    """
    root = Path(root).resolve()
    results, figures = root / 'results', root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    verification = verify_saved(root)
    cfg = json.loads((root / 'config.json').read_text())
    seeds = [int(s) for s in cfg['seeds']]
    assert len(seeds) == 3 and len(set(seeds)) == 3
    assert REPRESENTATIVE_SEED in seeds, 'Illustrative seed must be predeclared.'
    table = pd.read_csv(results / 'comparison.csv')
    expected = ['logistic'] + [f'{v}_{s}' for v in ('cnn', 'dropout') for s in seeds]
    assert set(table['run']) == set(expected) and len(table) == len(expected)
    table = table.set_index('run').loc[expected].reset_index()
    with np.load(results / 'test_predictions.npz') as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    y, indices = saved['y_true'], saved['test_index']
    assert np.array_equal(indices, np.arange(len(y)))

    summary_rows = []
    for variant in ('logistic', 'cnn', 'dropout'):
        rows = table[table['variant'] == variant]
        assert rows['parameters'].nunique() == 1
        row = {'variant': variant, 'runs': len(rows),
               'parameters': int(rows['parameters'].iloc[0])}
        for metric in ('accuracy', 'macro_f1', 'balanced_accuracy', 'train_seconds'):
            row[f'{metric}_mean'] = float(rows[metric].mean())
            row[f'{metric}_sd'] = float(rows[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results / 'summary.csv', index=False)

    pairs = []
    indexed = table.set_index('run')
    for seed in seeds:
        a, b = indexed.loc[f'cnn_{seed}'], indexed.loc[f'dropout_{seed}']
        pairs.append({'seed': seed, 'cnn_macro_f1': float(a['macro_f1']),
                      'dropout_macro_f1': float(b['macro_f1']),
                      'delta_macro_f1': float(b['macro_f1'] - a['macro_f1']),
                      'delta_macro_f1_percentage_points': float(100 * (b['macro_f1'] - a['macro_f1'])),
                      'cnn_accuracy': float(a['accuracy']),
                      'dropout_accuracy': float(b['accuracy']),
                      'delta_accuracy': float(b['accuracy'] - a['accuracy'])})
    pd.DataFrame(pairs).to_csv(results / 'matched_seed_deltas.csv', index=False)

    class_rows, error_rows, matrices = [], [], {}
    for run in expected:
        pred = saved[run]
        assert pred.shape == y.shape and np.isin(pred, np.arange(10)).all()
        cm = confusion_matrix(y, pred, labels=np.arange(10))
        matrices[run] = cm
        frame = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        frame.index.name = 'true_label'
        frame.to_csv(results / f'confusion_{run}.csv')
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=np.arange(10), zero_division=0)
        variant = indexed.loc[run, 'variant']
        seed = indexed.loc[run, 'seed']
        for label in range(10):
            class_rows.append({'run': run, 'variant': variant, 'seed': seed,
                               'class_id': label, 'class_name': LABELS[label],
                               'precision': float(precision[label]),
                               'recall': float(recall[label]), 'f1': float(f1[label]),
                               'support': int(support[label])})
            for predicted in range(10):
                if label != predicted:
                    error_rows.append({'run': run, 'variant': variant, 'seed': seed,
                                       'true_class_id': label, 'true_class': LABELS[label],
                                       'predicted_class_id': predicted,
                                       'predicted_class': LABELS[predicted],
                                       'count': int(cm[label, predicted]),
                                       'fraction_of_true_class': float(cm[label, predicted] / support[label])})
    classes = pd.DataFrame(class_rows)
    classes.to_csv(results / 'class_metrics.csv', index=False)
    errors = pd.DataFrame(error_rows).sort_values(
        ['run', 'count', 'true_class_id', 'predicted_class_id'],
        ascending=[True, False, True, True], kind='stable')
    errors.to_csv(results / 'error_pairs.csv', index=False)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    summary_index = summary.set_index('variant')
    for ax, metric, title in zip(axes, ('accuracy', 'macro_f1'), ('Accuracy', 'Macro-F1 (primary)')):
        for i, variant in enumerate(('logistic', 'cnn', 'dropout')):
            values = table.loc[table['variant'] == variant, metric].to_numpy()
            mean = summary_index.loc[variant, f'{metric}_mean']
            sd = summary_index.loc[variant, f'{metric}_sd']
            if len(values) > 1:
                ax.errorbar(i, mean, yerr=sd, fmt='D', color=COLORS[variant],
                            markersize=8, capsize=7, linewidth=2.2, zorder=3)
                jitter = np.linspace(-.16, .16, len(values))
                ax.scatter(i + jitter, values, s=36, facecolor='white',
                           edgecolor=COLORS[variant], linewidth=1.5, zorder=4)
            else:
                ax.scatter(i, mean, s=65, marker='D', color=COLORS[variant], zorder=3)
            ax.annotate(f'{100 * mean:.2f}%', (i, mean), xytext=(0, 19),
                        textcoords='offset points', ha='center', fontsize=10,
                        fontweight='bold', color=COLORS[variant])
        low = float(table[metric].min()) - .025
        high = min(1., float(table[metric].max()) + .035)
        ax.set_ylim(max(0, low), high)
        ax.set_xlim(-.5, 2.5)
        ax.set_xticks(range(3), ['Logistic\nregression', 'CNN', 'CNN +\ndropout'])
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis='y', zorder=0)
    fig.suptitle('Held-out Fashion-MNIST comparison', fontsize=15, fontweight='bold')
    fig.text(.5, .025, 'Diamonds: means; open circles: all three seeds; whiskers: sample SD (ddof = 1).\n'
             'Logistic regression is one validation-selected fit; all runs use the same official test examples.',
             ha='center', fontsize=9, color='#52657A')
    fig.tight_layout(rect=(0, .11, 1, .93))
    _save(fig, figures, 'comparison')

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    for column, variant in enumerate(('cnn', 'dropout')):
        for position, seed in enumerate(seeds):
            history = pd.read_csv(results / f'history_{variant}_{seed}.csv')
            color = SEED_COLORS[position]
            selected_epoch = int(indexed.loc[f'{variant}_{seed}', 'selected_epoch'])
            selected = history[history['epoch'] == selected_epoch].iloc[0]
            for row, train_key, val_key in ((0, 'train_loss_online', 'val_loss'),
                                           (1, 'train_macro_f1_online', 'val_macro_f1')):
                ax = axes[row, column]
                ax.plot(history['epoch'], history[train_key], '--', color=color, alpha=.8, linewidth=1.6)
                ax.plot(history['epoch'], history[val_key], '-', color=color, linewidth=1.9)
                ax.scatter([selected_epoch], [selected[val_key]], color=color, s=28, zorder=4)
                ax.grid(alpha=.8)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, column].set_title(DISPLAY[variant] + ': cross-entropy')
        axes[1, column].set_title(DISPLAY[variant] + ': macro-F1')
        axes[1, column].set_xlabel('Epoch')
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylabel('Cross-entropy')
    axes[1, 0].set_ylabel('Macro-F1')
    for row in (0, 1):
        limits = [ax.get_ylim() for ax in axes[row]]
        lo, hi = min(x[0] for x in limits), max(x[1] for x in limits)
        for ax in axes[row]:
            ax.set_ylim(lo, hi)
    handles = [Line2D([0], [0], color=SEED_COLORS[i], linewidth=2, label=f'Seed {s}')
               for i, s in enumerate(seeds)]
    handles.extend([Line2D([0], [0], color='#52657A', linestyle='--', label='Training (online)'),
                    Line2D([0], [0], color='#52657A', label='Validation (eval mode)'),
                    Line2D([0], [0], color='#52657A', marker='o', linestyle='', label='Selected checkpoint')])
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    fig.suptitle('Learning trajectories for every recorded seed', fontsize=15, fontweight='bold')
    fig.text(.5, .015, 'Online training metrics use evolving weights and active dropout; validation uses fixed epoch-end weights.\n'
             'These curves therefore do not measure an exact train–validation generalization gap.',
             ha='center', fontsize=9, color='#52657A')
    fig.tight_layout(rect=(0, .17, 1, .95))
    _save(fig, figures, 'learning_curves')

    representative = ['logistic', f'cnn_{REPRESENTATIVE_SEED}', f'dropout_{REPRESENTATIVE_SEED}']
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4))
    for ax, run in zip(axes, representative):
        cm = matrices[run]
        normalized = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7.4,
                            color='white' if normalized[i, j] > .55 else '#182B3A')
        ax.set_xticks(range(10), SHORT_LABELS, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(10), SHORT_LABELS, fontsize=8)
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')
        title = DISPLAY[indexed.loc[run, 'variant']]
        if run != 'logistic':
            title += f' (seed {REPRESENTATIVE_SEED})'
        ax.set_title(title)
        ax.tick_params(length=0)
    fig.suptitle('Where the models confuse clothing categories', fontsize=15, fontweight='bold')
    fig.text(.5, .015, 'Cell labels: image counts. Color: fraction within each true class (white = 0, dark blue = 1).\n'
             f'Seed {REPRESENTATIVE_SEED} was designated before testing; matrices for all seven runs are saved as CSV files.',
             ha='center', fontsize=9, color='#52657A')
    fig.tight_layout(rect=(0, .09, 1, .94))
    _save(fig, figures, 'confusion_matrices')

    # Compact single-column publication panel. Numeric IDs use the dataset's
    # published mapping, supplied alongside the figure in the report.
    run = f'dropout_{REPRESENTATIVE_SEED}'
    cm = matrices[run]
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7.5,
                        color='white' if normalized[i, j] > .55 else '#182B3A')
    ax.set_xticks(range(10), range(10), fontsize=8)
    ax.set_yticks(range(10), range(10), fontsize=8)
    ax.set_xlabel('Predicted class ID', fontsize=9)
    ax.set_ylabel('True class ID', fontsize=9)
    ax.set_title(f'CNN + dropout (seed {REPRESENTATIVE_SEED})', fontsize=10, pad=8)
    ax.tick_params(length=0)
    fig.tight_layout(pad=.55)
    _save(fig, figures, f'confusion_dropout_{REPRESENTATIVE_SEED}')

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    x = np.arange(10)
    for offset, variant in zip((-.22, 0., .22), ('logistic', 'cnn', 'dropout')):
        group = classes[classes['variant'] == variant].groupby('class_id')['f1']
        values = group.mean().reindex(range(10)).to_numpy()
        sd = group.std(ddof=1).reindex(range(10)).to_numpy()
        ax.errorbar(x + offset, values, yerr=None if variant == 'logistic' else sd,
                    fmt='o', color=COLORS[variant], markersize=5.5, capsize=3,
                    label=DISPLAY[variant], linewidth=1.5)
    ax.set_xticks(x, LABELS, rotation=35, ha='right')
    ax.set_ylim(max(0, classes['f1'].min() - .06), 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel('Per-class F1')
    ax.grid(axis='y')
    ax.set_title('Category-level performance on the official test set', pad=15)
    ax.legend(loc='lower right', frameon=False)
    fig.text(.5, .012, 'CNN markers show means across three seeds; whiskers show sample SD. Logistic regression has one fit.\n'
             'Class difficulty is an exploratory test-set observation, not a tuning signal.',
             ha='center', fontsize=9, color='#52657A')
    fig.tight_layout(rect=(0, .09, 1, 1))
    _save(fig, figures, 'class_f1')

    run = f'dropout_{REPRESENTATIVE_SEED}'
    selected_pairs = errors[(errors['run'] == run) & (errors['count'] > 0)].head(8)
    assert len(selected_pairs) == 8, 'Need eight nonempty directed error pairs.'
    images = read_idx(root / 'data/raw/t10k-images-idx3-ubyte.gz')
    assert images.shape == (len(y), 28, 28)
    chosen = []
    for row in selected_pairs.to_dict('records'):
        true, predicted = int(row['true_class_id']), int(row['predicted_class_id'])
        index = int(np.flatnonzero((y == true) & (saved[run] == predicted))[0])
        chosen.append({'test_index': index, 'run': run, 'true_class_id': true,
                       'true_class': LABELS[true], 'predicted_class_id': predicted,
                       'predicted_class': LABELS[predicted], 'pair_count': int(row['count'])})
    save_json(results / 'errors.json', {
        'analysis': 'Exploratory after test evaluation',
        'run': run, 'representative_seed_predeclared': REPRESENTATIVE_SEED,
        'selection_rule': 'Eight largest directed off-diagonal counts; ties by true then predicted class ID. '
                          'For each pair choose the lowest official test index.',
        'examples': chosen,
    })
    fig, axes = plt.subplots(2, 4, figsize=(11, 7))
    for ax, example in zip(axes.flat, chosen):
        ax.imshow(images[example['test_index']], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"True: {example['true_class']}\nPredicted: {example['predicted_class']}", fontsize=10, pad=7)
        ax.set_xlabel(f"Test index {example['test_index']} | pair errors: {example['pair_count']}", fontsize=8)
    fig.suptitle(f'Exploratory errors: CNN + dropout, predeclared seed {REPRESENTATIVE_SEED}',
                 fontsize=14, fontweight='bold')
    fig.text(.5, .018, 'First test image in each of the eight most frequent directed confusion pairs; indices are zero-based.\n'
             'Selection emphasizes common failures and does not represent the prevalence of all image appearances.',
             ha='center', fontsize=9, color='#52657A')
    fig.tight_layout(rect=(0, .075, 1, .94), h_pad=2.0)
    _save(fig, figures, 'representative_errors')

    return {'verification': verification, 'summary': str(results / 'summary.csv'),
            'matched_seed_deltas': str(results / 'matched_seed_deltas.csv'),
            'class_metrics': str(results / 'class_metrics.csv'),
            'error_pairs': str(results / 'error_pairs.csv'), 'errors': str(results / 'errors.json'),
            'figures': [str(figures / name) for name in
                        ('comparison.pdf', 'learning_curves.pdf', 'confusion_matrices.pdf',
                         f'confusion_dropout_{REPRESENTATIVE_SEED}.pdf',
                         'class_f1.pdf', 'representative_errors.pdf')]}


if __name__ == '__main__':
    print(json.dumps(build_analysis(), indent=2))
