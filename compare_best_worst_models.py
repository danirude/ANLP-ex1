import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np


def compare_validation_predictions(
    best_model_dir: str,
    worst_model_dir: str,
    out_dir: str = "validation_best_worst_models_predictions_comparisons_outputs"
) -> tuple[Path, Path, Path]:
    """
    Compare validation‑prediction Excel files of the best and worst models.

    Saves:
      • full comparison dataset
      • subset where best == correct & worst == wrong
      • combined PNG confusion‑matrix heat‑maps (percentages)
    Returns the three paths in that order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 ── Load files ───────────────────────────────────────────────────────────
    best_excel  = Path(best_model_dir)  / "validation_predictions.xlsx"
    worst_excel = Path(worst_model_dir) / "validation_predictions.xlsx"
    best  = pd.read_excel(best_excel)
    worst = pd.read_excel(worst_excel)

    best  = best.rename(columns={"predicted_label": "best_pred",
                                 "correct":         "best_correct"})
    worst = worst.rename(columns={"predicted_label": "worst_pred",
                                  "correct":         "worst_correct"})

    # 2 ── Merge on sentence pair ──────────────────────────────────────────────
    merged = pd.merge(
        best[["sentence1", "sentence2", "true_label", "best_pred", "best_correct"]],
        worst[["sentence1", "sentence2", "worst_pred", "worst_correct"]],
        on=["sentence1", "sentence2"],
        how="inner",
        validate="one_to_one"
    )

    # 3 ── Save full comparison ────────────────────────────────────────────────
    full_path = out_dir / "best_worst_models_comparisons.xlsx"
    merged.to_excel(full_path, index=False)

    # 4 ── Subset where best correct & worst wrong ─────────────────────────────
    subset = merged[(merged["best_correct"] == 1) & (merged["worst_correct"] == 0)]
    subset_path = out_dir / "best_correct_worst_wrong.xlsx"
    subset.to_excel(subset_path, index=False)

    # 5 ── Combined Confusion Matrices (percentages) ───────────────────────────
    def plot_combined_heatmaps(true, best_pred, worst_pred, labels, file_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        titles = ["Best Model – Confusion Matrix (%)", "Worst Model – Confusion Matrix (%)"]
        preds = [best_pred, worst_pred]
        cmaps = ["Blues", "Reds"]

        for ax, pred, title, cmap in zip(axes, preds, titles, cmaps):
            cm = confusion_matrix(true, pred, labels=labels)
            cm_percent = cm.astype(np.float64) / cm.sum() * 100

            sns.heatmap(
                cm_percent.round(1),
                annot=True,
                fmt=".1f",
                cmap=cmap,
                cbar_kws={"label": "% of total"},
                xticklabels=["Not Paraphrase", "Paraphrase"],
                yticklabels=["Not Paraphrase", "Paraphrase"],
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    combined_cm_path = out_dir / "combined_confusion_matrices_pct.png"

    plot_combined_heatmaps(
        merged["true_label"],
        merged["best_pred"],
        merged["worst_pred"],
        labels=[0, 1],
        file_path=combined_cm_path
    )

    print(f"✓ Full comparison written to       {full_path}")
    print(f"✓ Subset (best‑right / worst‑wrong) {subset_path}")
    print(f"✓ Combined confusion matrix         {combined_cm_path}")

    return full_path, subset_path, combined_cm_path


# ── Example usage ─────────────────────────────────────────────────────────────
best_model_dir = "epoch_num_3_lr_0.0001_batch_size_32"
worst_model_dir = "epoch_num_5_lr_0.0005_batch_size_32"

compare_validation_predictions(best_model_dir, worst_model_dir)
