import os, shutil, zipfile
from pathlib import Path

# ── CONFIG — update these paths before running ────────────────────────────────
from pathlib import Path
_REPO_ROOT   = Path(__file__).parent.parent.resolve()
DATASET_ZIP  = str(_REPO_ROOT / "databank" / "contam" / "combined_training_data.zip")
DATASET_DIR  = str(_REPO_ROOT / "databank" / "contam" / "combined_training_data")
RUNS_DIR     = str(_REPO_ROOT / "training_results" / "Contamination" / "Long2-Data" / "runs")
DRIVE_OUTPUT = str(_REPO_ROOT / "training_results" / "Contamination" / "Long2-Data" / "results")
EPOCHS       = 150
IMG_SIZE     = 224
BATCH        = 16
CLASS_NAMES  = ['Contaminant', 'No_Contaminant']
MODEL_11     = 'yolo11s'
MODEL_26     = 'yolo26s'
COLORS       = {MODEL_11: '#4C72B0', MODEL_26: '#DD8452'}
CLEAR_RUNS   = True
SKIP_TRAINING = False

def setup_dataset(dataset_zip, dataset_dir):
    """Extract dataset if needed and print image counts per split/class."""
    if not os.path.exists(dataset_dir):
        print('Extracting dataset...')
        with zipfile.ZipFile(dataset_zip, 'r') as z:
            z.extractall(Path(dataset_dir).parent)
        print('Done.')
    else:
        print('Dataset already extracted.')

    for split in ['train', 'val', 'test']:
        for cls in CLASS_NAMES:
            p = Path(dataset_dir) / split / cls
            n = len(list(p.glob('*.png'))) if p.exists() else 0
            print(f'  {split:5s} / {cls:16s}: {n} images')

def clear_old_runs(clear_runs: bool):
    if clear_runs:
        for run in [MODEL_11, MODEL_26]:
            run_path = Path(RUNS_DIR) / run
            if run_path.exists():
                shutil.rmtree(run_path)
                print(f'Cleared old run: {run_path}')

def train_model(model_name, model_file, dataset_dir, runs_dir):
    """Train a single YOLO classification model and return the results."""
    from ultralytics import YOLO
    import gc, torch
    print('=' * 60)
    print(f'Training {model_name}')
    print('=' * 60)
    model = YOLO(model_file)
    results = model.train(
        data=dataset_dir,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=runs_dir,
        name=model_name.lower(),
        exist_ok=False,
        plots=True,
    )
    print(f'{model_name} training complete!')

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f'GPU memory cleared.')
    return results

def evaluate_model(model_path, model_name, test_dir):
    """Run inference on test set and return ground truth, predictions, confidences."""
    from ultralytics import YOLO
    from sklearn.metrics import classification_report
    import numpy as np

    model = YOLO(model_path)
    y_true, y_pred, y_conf = [], [], []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(test_dir) / class_name
        images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        for img_path in images:
            result    = model(str(img_path), verbose=False)[0]
            probs     = result.probs
            y_true.append(class_idx)
            y_pred.append(int(probs.top1))
            y_conf.append(float(probs.top1conf))

    print(f'\n{"=" * 50}')
    print(f'  {model_name} -- Test Set Results')
    print(f'{"=" * 50}')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def load_results_csv(runs_dir, run_name):
    """Load a YOLO results.csv and strip column name whitespace."""
    import pandas as pd
    p = Path(runs_dir) / run_name / 'results.csv'
    if p.exists():
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        return df
    return None


def plot_training_curves(df_11, df_26, plot_dir):
    """Plot train loss, val loss, and val top-1 accuracy for both models."""
    import matplotlib.pyplot as plt
    import numpy as np

    ref  = df_11 if df_11 is not None else df_26
    cols = list(ref.columns) if ref is not None else []
    LOSS_COL     = next((c for c in cols if 'train' in c.lower() and 'loss' in c.lower()), None)
    VAL_LOSS_COL = next((c for c in cols if 'val'   in c.lower() and 'loss' in c.lower()), None)
    ACC_COL      = next((c for c in cols if 'top1'  in c.lower() or ('acc' in c.lower() and 'val' in c.lower())), None)
    print(f'train_loss={LOSS_COL}  val_loss={VAL_LOSS_COL}  accuracy={ACC_COL}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')

    for df, label, color in [(df_11, MODEL_11, COLORS[MODEL_11]),
                              (df_26, MODEL_26, COLORS[MODEL_26])]:
        if df is None:
            continue
        ep = df['epoch'] if 'epoch' in df.columns else range(len(df))
        if LOSS_COL     and LOSS_COL     in df.columns: axes[0].plot(ep, df[LOSS_COL],     label=label, color=color, lw=2)
        if VAL_LOSS_COL and VAL_LOSS_COL in df.columns: axes[1].plot(ep, df[VAL_LOSS_COL], label=label, color=color, lw=2)
        if ACC_COL      and ACC_COL      in df.columns: axes[2].plot(ep, df[ACC_COL],      label=label, color=color, lw=2)

    axes[0].set(title='Train Loss',         xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Val Loss',           xlabel='Epoch', ylabel='Loss')
    axes[2].set(title='Val Top-1 Accuracy', xlabel='Epoch', ylabel='Accuracy')
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'training_curves.png', bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(y_true_11, y_pred_11, y_true_26, y_pred_26, plot_dir):
    """Plot confusion matrices for both models side by side."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices - Test Set', fontsize=14, fontweight='bold')

    for ax, y_true, y_pred, title in [
        (axes[0], y_true_11, y_pred_11, MODEL_11),
        (axes[1], y_true_26, y_pred_26, MODEL_26),
    ]:
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(plot_dir / 'confusion_matrices.png', bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(y_true_11, y_pred_11, y_true_26, y_pred_26, plot_dir):
    """Plot per-class precision, recall, and F1 for both models."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    def get_metrics(y_true, y_pred):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
        return {'Precision': p, 'Recall': r, 'F1': f}

    m11  = get_metrics(y_true_11, y_pred_11)
    m26  = get_metrics(y_true_26, y_pred_26)
    x    = np.arange(len(CLASS_NAMES))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Per-Class Metrics - Test Set', fontsize=14, fontweight='bold')

    for i, metric in enumerate(['Precision', 'Recall', 'F1']):
        ax = axes[i]
        b1 = ax.bar(x - width/2, m11[metric], width, label=MODEL_11, color=COLORS[MODEL_11], alpha=0.85)
        b2 = ax.bar(x + width/2, m26[metric], width, label=MODEL_26, color=COLORS[MODEL_26], alpha=0.85)
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=10, ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_dir / 'per_class_metrics.png', bbox_inches='tight')
    plt.show()


def plot_confidence_distributions(y_true_11, y_pred_11, y_conf_11,
                                   y_true_26, y_pred_26, y_conf_26, plot_dir):
    """Plot prediction confidence distributions for both models."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Prediction Confidence Distributions - Test Set', fontsize=14, fontweight='bold')

    for ax, y_true, y_pred, y_conf, title in [
        (axes[0], y_true_11, y_pred_11, y_conf_11, MODEL_11),
        (axes[1], y_true_26, y_pred_26, y_conf_26, MODEL_26),
    ]:
        bins      = np.linspace(0, 1, 21)
        correct   = y_conf[y_true == y_pred]
        incorrect = y_conf[y_true != y_pred]
        ax.hist(correct,   bins=bins, alpha=0.7, label='Correct',   color='#2ca02c')
        ax.hist(incorrect, bins=bins, alpha=0.7, label='Incorrect', color='#d62728')
        ax.axvline(np.mean(y_conf), color='black', linestyle='--', lw=1.5,
                   label=f'Mean: {np.mean(y_conf):.2f}')
        ax.set(title=title, xlabel='Confidence', ylabel='Count')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'confidence_distributions.png', bbox_inches='tight')
    plt.show()


def plot_model_comparison(y_true_11, y_pred_11, y_true_26, y_pred_26, plot_dir):
    """Plot overall accuracy and F1 comparison and return summary DataFrame."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    summary = {
        'Model':            [MODEL_11, MODEL_26],
        'Accuracy':         [accuracy_score(y_true_11, y_pred_11),
                             accuracy_score(y_true_26, y_pred_26)],
        'F1 (macro)':       [f1_score(y_true_11, y_pred_11, average='macro'),
                             f1_score(y_true_26, y_pred_26, average='macro')],
        'F1 (Contaminant)': [f1_score(y_true_11, y_pred_11, pos_label=0),
                             f1_score(y_true_26, y_pred_26, pos_label=0)],
    }
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    metrics = ['Accuracy', 'F1 (macro)', 'F1 (Contaminant)']
    x       = np.arange(len(metrics))
    width   = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, color) in enumerate([(MODEL_11, COLORS[MODEL_11]),
                                         (MODEL_26, COLORS[MODEL_26])]):
        vals = [df_summary.loc[i, m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Overall Model Comparison - Test Set', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'model_comparison_summary.png', bbox_inches='tight')
    plt.show()

    return df_summary


def plot_test_image_grid(model_path, model_name, test_dir, plot_dir, max_images=40):
    """Display test images in a grid with green/red borders for correct/incorrect."""
    from ultralytics import YOLO
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    model   = YOLO(model_path)
    records = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(test_dir) / class_name
        images    = sorted(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
        for img_path in images:
            result    = model(str(img_path), verbose=False)[0]
            probs     = result.probs
            pred_idx  = int(probs.top1)
            pred_conf = float(probs.top1conf)
            correct   = pred_idx == class_idx
            records.append({
                'path':    img_path,
                'true':    class_name,
                'pred':    CLASS_NAMES[pred_idx],
                'conf':    pred_conf,
                'correct': correct,
            })

    incorrect = [r for r in records if not r['correct']]
    correct   = [r for r in records if r['correct']]
    sample    = incorrect + correct[:min(max_images - len(incorrect), len(correct))]
    sample.sort(key=lambda r: (r['correct'], r['true']))

    ncols = 8
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.5))
    fig.suptitle(
        f'{model_name} — Test Set Predictions\n'
        f'(green = correct, red = incorrect | caption: pred / true / conf)',
        fontsize=13, fontweight='bold', y=1.01
    )
    axes = np.array(axes).flatten()

    for ax, rec in zip(axes, sample):
        img          = Image.open(rec['path']).convert('RGB')
        border_color = '#2ca02c' if rec['correct'] else '#d62728'
        border       = 6
        w, h         = img.size
        bordered     = Image.new('RGB', (w + border*2, h + border*2), border_color)
        bordered.paste(img, (border, border))
        ax.imshow(bordered)
        p = 'Cont' if rec['pred'] == 'Contaminant' else 'NoCont'
        t = 'Cont' if rec['true'] == 'Contaminant' else 'NoCont'
        ax.set_title(f'{p} / {t}\n{rec["conf"]:.2f}', fontsize=7.5,
                     color=border_color, fontweight='bold')
        ax.axis('off')

    for ax in axes[len(sample):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name.lower()}_test_predictions.png', bbox_inches='tight', dpi=120)
    plt.show()

    total     = len(records)
    n_correct = sum(r['correct'] for r in records)
    print(f'\n{model_name} test accuracy: {n_correct}/{total} = {n_correct/total:.1%}')
    print(f'Misclassified: {total - n_correct}')
    return records


def plot_errors_comparison(records_11, records_26, plot_dir):
    """Show images misclassified by either model, with both predictions side by side."""
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    lookup_26 = {str(r['path']): r for r in records_26}
    errors    = [
        r for r in records_11
        if not r['correct'] or not lookup_26.get(str(r['path']), {}).get('correct', True)
    ]

    if not errors:
        print('Both models correctly classified all test images!')
        return

    ncols = 6
    nrows = math.ceil(len(errors) / (ncols // 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.8))
    fig.suptitle(
        'Misclassified Images (by either model)\n'
        'Left = ' + MODEL_11 +'   |   Right = ' + MODEL_26 + '\n'
        'Caption: pred / true / conf',
        fontsize=12, fontweight='bold', y=1.02
    )
    axes = np.array(axes).reshape(-1, ncols)

    def render(ax, rec, model_label):
        img          = Image.open(rec['path']).convert('RGB')
        border_color = '#2ca02c' if rec['correct'] else '#d62728'
        border       = 6
        w, h         = img.size
        bordered     = Image.new('RGB', (w + border*2, h + border*2), border_color)
        bordered.paste(img, (border, border))
        ax.imshow(bordered)
        p = 'Cont' if rec['pred'] == 'Contaminant' else 'NoCont'
        t = 'Cont' if rec['true'] == 'Contaminant' else 'NoCont'
        ax.set_title(f'{model_label}\n{p}/{t} {rec["conf"]:.2f}',
                     fontsize=7, color=border_color, fontweight='bold')
        ax.axis('off')

    col, row = 0, 0
    for rec_11 in errors:
        rec_26 = lookup_26.get(str(rec_11['path']), rec_11)
        if col + 2 > ncols:
            col = 0
            row += 1
        if row < nrows:
            render(axes[row, col],     rec_11, MODEL_11)
            render(axes[row, col + 1], rec_26, MODEL_26)
        col += 2

    for r in range(nrows):
        for c in range(ncols):
            if not axes[r, c].has_data():
                axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(plot_dir / 'misclassified_comparison.png', bbox_inches='tight', dpi=120)
    plt.show()


def save_results(runs_dir, plot_dir, df_summary, output_dir):
    """Copy plots and best weights to the output directory and save summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(str(plot_dir), f'{output_dir}/comparison_plots', dirs_exist_ok=True)

    for run in [MODEL_11, MODEL_26]:
        src = Path(runs_dir) / run / 'weights' / 'best.pt'
        dst = Path(output_dir) / f'{run}_best.pt'
        if src.exists():
            shutil.copy2(src, dst)
            print(f'Saved: {dst}')

    df_summary.to_csv(f'{output_dir}/test_results_summary.csv', index=False)
    print(f'All results saved to: {output_dir}')

def save_excel_report(runs_dir, y_true_11, y_pred_11, y_conf_11,
                       y_true_26, y_pred_26, y_conf_26, output_dir):
    """Save all training and evaluation data to an Excel file with multiple sheets."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    from pathlib import Path

    output_path = Path(output_dir) / 'results.xlsx'
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # ── Sheet 1: Overall summary ──────────────────────────────────────
    summary = {
        'Model':            [MODEL_11, MODEL_26],
        'Accuracy':         [accuracy_score(y_true_11, y_pred_11),
                             accuracy_score(y_true_26, y_pred_26)],
        'F1 (macro)':       [f1_score(y_true_11, y_pred_11, average='macro'),
                             f1_score(y_true_26, y_pred_26, average='macro')],
        'F1 (Contaminant)': [f1_score(y_true_11, y_pred_11, pos_label=0),
                             f1_score(y_true_26, y_pred_26, pos_label=0)],
        'F1 (No_Contaminant)': [f1_score(y_true_11, y_pred_11, pos_label=1),
                                 f1_score(y_true_26, y_pred_26, pos_label=1)],
        'Mean Confidence':  [np.mean(y_conf_11), np.mean(y_conf_26)],
        'Epochs':           [EPOCHS, EPOCHS],
        'Img Size':         [IMG_SIZE, IMG_SIZE],
        'Batch Size':       [BATCH, BATCH],
    }
    pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    # ── Sheet 2: Per-class metrics ────────────────────────────────────
    rows = []
    for model_name, y_true, y_pred in [(MODEL_11, y_true_11, y_pred_11),
                                        (MODEL_26, y_true_26, y_pred_26)]:
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
        for i, cls in enumerate(['Contaminant', 'No_Contaminant']):
            rows.append({
                'Model':     model_name,
                'Class':     cls,
                'Precision': p[i],
                'Recall':    r[i],
                'F1':        f[i],
                'Support':   s[i],
            })
    pd.DataFrame(rows).to_excel(writer, sheet_name='Per-Class Metrics', index=False)

    # ── Sheet 3: Training curves (epoch-by-epoch) ─────────────────────
    for run_name, label in [(MODEL_11, MODEL_11), (MODEL_26, MODEL_26)]:
        p = Path(runs_dir) / run_name / 'results.csv'
        if p.exists():
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            df.insert(0, 'Model', label)
            sheet = f'Training Curves {label}'
            df.to_excel(writer, sheet_name=sheet, index=False)

    # ── Sheet 4: Per-image predictions ───────────────────────────────
    pred_rows = []
    for model_name, y_true, y_pred, y_conf in [
        (MODEL_11, y_true_11, y_pred_11, y_conf_11),
        (MODEL_26, y_true_26, y_pred_26, y_conf_26),
    ]:
        for i, (yt, yp, yc) in enumerate(zip(y_true, y_pred, y_conf)):
            pred_rows.append({
                'Model':      model_name,
                'Image Index': i,
                'True Label': CLASS_NAMES[yt],
                'Pred Label': CLASS_NAMES[yp],
                'Confidence': yc,
                'Correct':    yt == yp,
            })
    pd.DataFrame(pred_rows).to_excel(writer, sheet_name='Per-Image Predictions', index=False)

    writer.close()
    print(f'Excel report saved to: {output_path}')

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # 1. Setup
    setup_dataset(DATASET_ZIP, DATASET_DIR)

    # Clear old runs before training
    clear_old_runs(CLEAR_RUNS)
    # 2. Train
    if not SKIP_TRAINING:
        train_model(MODEL_26, MODEL_26+'-cls.pt', DATASET_DIR, RUNS_DIR)
        train_model(MODEL_11, MODEL_11+'-cls.pt', DATASET_DIR, RUNS_DIR)

    # 3. Evaluate
    TEST_DIR = Path(DATASET_DIR) / 'test'
    y_true_11, y_pred_11, y_conf_11 = evaluate_model(Path(RUNS_DIR) / MODEL_11 / 'weights' / 'best.pt', MODEL_11, TEST_DIR)
    y_true_26, y_pred_26, y_conf_26 = evaluate_model(Path(RUNS_DIR) / MODEL_26 / 'weights' / 'best.pt', MODEL_26, TEST_DIR)

    # 4. Plots
    PLOT_DIR = Path(RUNS_DIR) / 'comparison_plots'
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df_11 = load_results_csv(RUNS_DIR, MODEL_11)
    df_26 = load_results_csv(RUNS_DIR, MODEL_26)

    plot_training_curves(df_11, df_26, PLOT_DIR)
    plot_confusion_matrices(y_true_11, y_pred_11, y_true_26, y_pred_26, PLOT_DIR)
    plot_per_class_metrics(y_true_11, y_pred_11, y_true_26, y_pred_26, PLOT_DIR)
    plot_confidence_distributions(y_true_11, y_pred_11, y_conf_11, y_true_26, y_pred_26, y_conf_26, PLOT_DIR)
    df_summary = plot_model_comparison(y_true_11, y_pred_11, y_true_26, y_pred_26, PLOT_DIR)

    records_11 = plot_test_image_grid(Path(RUNS_DIR) / MODEL_11 / 'weights' / 'best.pt', MODEL_11, TEST_DIR, PLOT_DIR)
    records_26 = plot_test_image_grid(Path(RUNS_DIR) / MODEL_26 / 'weights' / 'best.pt', MODEL_26, TEST_DIR, PLOT_DIR)
    plot_errors_comparison(records_11, records_26, PLOT_DIR)

    # 5. Save
    save_results(RUNS_DIR, PLOT_DIR, df_summary, DRIVE_OUTPUT)
    save_excel_report(RUNS_DIR, y_true_11, y_pred_11, y_conf_11,
                      y_true_26, y_pred_26, y_conf_26, DRIVE_OUTPUT)