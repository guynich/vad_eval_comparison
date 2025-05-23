"""Evaluate VAD model performance using AUC curves with Hugging Face dataset."""

import pprint
from dataclasses import dataclass
from typing import List

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
from datasets import DatasetDict, disable_caching, load_dataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from speech_detector_silero_vad import SpeechDetector as SpeechDetectorSileroVAD
from speech_detector_ten_vad import SpeechDetector as SpeechDetectorTenVAD

SHOW_ALL_PLOTS = False  # When True plot AUC curves for all splits.


@dataclass
class AUCMetrics:
    """Area Under Curve metrics."""

    roc_auc: float
    pr_auc: float


def compute_overall_auc(
    y_true: List[List[int]], y_scores: List[List[float]]
) -> AUCMetrics:
    """Compute ROC and PR AUC scores for flattened predictions."""
    flat_true = np.concatenate(y_true)
    flat_scores = np.concatenate(y_scores)

    return AUCMetrics(
        roc_auc=roc_auc_score(flat_true, flat_scores),
        pr_auc=average_precision_score(flat_true, flat_scores),
    )


def plot_performance_curves(
    y_true: List[List[int]],
    y_scores: List[List[float]],
    model_name: str = "",
    split: str = "",
    confidence_label: str = "",
    threshold_markers: List[float] = [0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
) -> None:
    """Plot ROC and PR curves with threshold markers."""
    flat_true = np.concatenate(y_true)
    flat_scores = np.concatenate(y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(flat_true, flat_scores)
    roc_auc = roc_auc_score(flat_true, flat_scores)

    ax1.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", label="Random")
    _add_threshold_markers(ax1, fpr, tpr, roc_thresholds, threshold_markers)

    ax1.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{model_name}  {split}: ROC Curve{confidence_label}",
    )
    ax1.grid()
    ax1.legend()

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(flat_true, flat_scores)
    pr_auc = average_precision_score(flat_true, flat_scores)

    ax2.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}", lw=2)
    _add_threshold_markers(ax2, recall, precision, pr_thresholds, threshold_markers)

    ax2.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{model_name}  {split}: Precision-Recall Curve{confidence_label}",
        ylim=[0.75, 1.02],
    )
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show(block=False)


def plot_comparison_curves(
    model_results: List[tuple[List[List[int]], List[List[float]], str]],
    split: str = "",
    confidence_label: str = "",
    threshold_markers: List[float] = [0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
) -> None:
    """Plot ROC and PR curves for multiple models on same axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for y_true, y_scores, model_name in model_results:
        flat_true = np.concatenate(y_true)
        flat_scores = np.concatenate(y_scores)

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(flat_true, flat_scores)
        roc_auc = roc_auc_score(flat_true, flat_scores)

        line = ax1.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", lw=2)[0]
        _add_threshold_markers(
            ax1, fpr, tpr, roc_thresholds, threshold_markers, color=line.get_color()
        )

        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(
            flat_true, flat_scores
        )
        pr_auc = average_precision_score(flat_true, flat_scores)

        line = ax2.plot(
            recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", lw=2
        )[0]
        _add_threshold_markers(
            ax2,
            recall,
            precision,
            pr_thresholds,
            threshold_markers,
            color=line.get_color(),
        )

    # Add random baseline to ROC plot
    ax1.plot([0, 1], [0, 1], "k--", label="Random")

    ax1.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve Comparison - {split}{confidence_label}",
    )
    ax1.grid()
    ax1.legend()

    ax2.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall Curve Comparison - {split}{confidence_label}",
        ylim=[0.75, 1.02],
    )
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show(block=False)


def _add_threshold_markers(
    ax: plt.Axes,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    thresholds: np.ndarray,
    target_thresholds: List[float],
    color: str = "black",
) -> None:
    """Add threshold markers to plot."""
    for target in target_thresholds:
        idx = np.abs(thresholds - target).argmin()
        ax.plot(x_coords[idx], y_coords[idx], "x", color=color)
        ax.annotate(
            f"{thresholds[idx]:.2f}",
            (x_coords[idx], y_coords[idx]),
            fontsize=8,
            color=color,
            ha="right",
        )


def process_audio(audio: np.ndarray, detector: SpeechDetectorSileroVAD) -> List[float]:
    """Process audio chunks and return VAD probabilities."""
    detector.reset()
    chunk_size = detector.chunk_size
    return [
        detector(audio[i : i + chunk_size])
        for i in range(0, (len(audio) // chunk_size) * chunk_size, chunk_size)
    ]


def process_batch(examples: dict, detector: SpeechDetectorSileroVAD) -> dict:
    """Process a batch of audio examples."""
    vad_probs = [process_audio(audio["array"], detector) for audio in examples["audio"]]

    all_speech = examples["speech"]
    all_confidence = examples["confidence"]

    # Create masks for confident examples
    confident_masks = [np.array(conf) == 1 for conf in all_confidence]

    # Apply masks to get confident examples
    confident_speech = [
        np.array(speech)[mask] for speech, mask in zip(all_speech, confident_masks)
    ]
    confident_vad_probs = [
        np.array(probs)[mask] for probs, mask in zip(vad_probs, confident_masks)
    ]

    return {
        "vad_probs": vad_probs,
        "confident_speech": confident_speech,
        "confident_vad_probs": confident_vad_probs,
    }


def main():
    """Combine two datasets for the comparison."""
    disable_caching()  # Mitigate hashing failure.
    dataset_names = [
        "guynich/librispeech_asr_test_vad",
        "guynich/multilingual_librispeech_test_vad",
    ]

    # Load all subset test splits
    splits = []
    test_splits = {}

    for dataset_name in dataset_names:
        ds = load_dataset(
            dataset_name,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
            trust_remote_code=True,
        )
        for split_name, split_dataset in ds.items():
            splits.append(split_name)
            key = f"{split_name}"
            test_splits[key] = split_dataset

    dataset = DatasetDict(test_splits)

    silero_vad = SpeechDetectorSileroVAD()
    ten_vad = SpeechDetectorTenVAD()
    models = [silero_vad, ten_vad]

    auc_metric_results = {}

    # Add these before the try block
    all_splits_results = []
    all_splits_results_confident = []

    try:
        for split in splits:
            model_results = []
            model_results_confident = []

            for model in models:
                print(f"Processing {split} with {model.get_name()} ...")

                # Process dataset with batching
                processed = dataset[split].map(
                    lambda x: process_batch(x, model),
                    batched=True,
                    batch_size=64,
                    remove_columns=dataset[split].column_names,
                    load_from_cache_file=False,
                )

                # Store results for plotting
                model_results.append(
                    (
                        dataset[split]["speech"],
                        processed["vad_probs"],
                        f"{model.get_name()} ({split})",
                    )
                )
                model_results_confident.append(
                    (
                        processed["confident_speech"],
                        processed["confident_vad_probs"],
                        f"{model.get_name()} ({split})",
                    )
                )

                # Store for all-splits comparison
                all_splits_results.append(
                    (
                        dataset[split]["speech"],
                        processed["vad_probs"],
                        f"{model.get_name()} ({split})",
                    )
                )
                all_splits_results_confident.append(
                    (
                        processed["confident_speech"],
                        processed["confident_vad_probs"],
                        f"{model.get_name()} ({split})",
                    )
                )

                # Store metrics
                auc_metric_results[f"{split}_{model.get_name()}"] = compute_overall_auc(
                    dataset[split]["speech"], processed["vad_probs"]
                )
                auc_metric_results[f"{split}_{model.get_name()}_confidence"] = (
                    compute_overall_auc(
                        processed["confident_speech"], processed["confident_vad_probs"]
                    )
                )

            if SHOW_ALL_PLOTS:
                # Individual split comparisons
                plot_comparison_curves(model_results, split=split)
                plot_comparison_curves(
                    model_results_confident,
                    split=split,
                    confidence_label=" (exclude low confidence)",
                )

        if SHOW_ALL_PLOTS:
            # After processing all splits, plot the combined comparison
            plot_comparison_curves(all_splits_results, split="All Splits")
            plot_comparison_curves(
                all_splits_results_confident,
                split="All Splits",
                confidence_label=" (exclude low confidence)",
            )

        # Combine results by model
        model_combined_results = {}
        model_combined_results_confident = {}

        for y_true, y_scores, model_name in all_splits_results:
            model_name_base = model_name.split(" (")[0]  # Extract base model name
            if model_name_base not in model_combined_results:
                model_combined_results[model_name_base] = {
                    "y_true": [],
                    "y_scores": [],
                }
            model_combined_results[model_name_base]["y_true"].extend(y_true)
            model_combined_results[model_name_base]["y_scores"].extend(y_scores)

        for y_true, y_scores, model_name in all_splits_results_confident:
            model_name_base = model_name.split(" (")[0]
            if model_name_base not in model_combined_results_confident:
                model_combined_results_confident[model_name_base] = {
                    "y_true": [],
                    "y_scores": [],
                }
            model_combined_results_confident[model_name_base]["y_true"].extend(y_true)
            model_combined_results_confident[model_name_base]["y_scores"].extend(
                y_scores
            )

        # Create combined results lists for plotting
        combined_results = [
            (data["y_true"], data["y_scores"], model_name)
            for model_name, data in model_combined_results.items()
        ]

        combined_results_confident = [
            (data["y_true"], data["y_scores"], model_name)
            for model_name, data in model_combined_results_confident.items()
        ]

        # Plot combined results
        plot_comparison_curves(combined_results, split="All Data (Combined)")
        plot_comparison_curves(
            combined_results_confident,
            split="All Data (Combined)",
            confidence_label=" (exclude low confidence)",
        )

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")

    print("\nOverall results:")
    pprint.pprint(auc_metric_results)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
