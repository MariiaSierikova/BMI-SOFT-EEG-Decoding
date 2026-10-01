"""Compare non-overlapping and overlapping raw EEG predictions on one MOVING person."""

import argparse
from pathlib import Path

import numpy as np

from scripts.evaluation.recog_eval_data import (
    load_recording,
    make_batches,
    split_trials,
)
from scripts.evaluation.recog_eval_metrics import report_predictions, write_predictions
from scripts.treeoftrees.features import get_features
from scripts.treeoftrees.moving_data import DATA
from scripts.treeoftrees.tree_of_trees import TreeOfTrees


# ================================================================
# 1. Train the TreeOfTrees on non-overlapping batches
# ================================================================
# Keep only the clean batches: fully inside a movement or a rest.
# Extract the 224 EEG features (7 features x 32 channels) of each batch.
# Train the TreeOfTrees on them and print how well it knows its training batches.
def get_model(train_batches, train_infos):
    clean = np.asarray([info.clean for info in train_infos])
    labels = np.asarray([info.truth for info in train_infos])[clean]
    features = get_features(train_batches[clean])
    model = TreeOfTrees().fit(features, labels)
    print(f"Training accuracy: {(model.predict(features) == labels).mean():.1%}")
    return model

# ================================================================
# 2. Evaluate both window steps with the same model
# ================================================================
# Load one EDF and split trials by movement code.
# Train a model using non-overlapping training batches.
# Test with a full-window step and/or overlap_step_ms.
# Compare accuracy, recognition, and false-gesture rates.
# Save predictions to CSV if a path is provided.
def evaluate(
    edf: Path,
    window_ms: int = 1000,
    overlap_step_ms: int = 100,
    mode: str = "both",
    train_per_code: int = 4,
    csv_path: Path | None = None,
) -> None:
    if not edf.is_file():
        raise FileNotFoundError(edf)
    if window_ms <= 0 or train_per_code <= 0:
        raise ValueError("window_ms and train_per_code must be positive")
    if mode in ("both", "overlap") and not 0 < overlap_step_ms < window_ms:
        raise ValueError("Overlapping step must be between 0 and window size")

    eeg, timestamps, sfreq, trials, bounds = load_recording(edf)
    train_trials, test_trials = split_trials(trials, train_per_code)
    train_batches, train_infos = make_batches(
        eeg, timestamps, sfreq, trials, bounds, train_trials, window_ms, window_ms
    )
    model = get_model(train_batches, train_infos)
    del train_batches

    modes = []
    if mode in ("both", "non-overlap"):
        modes.append(("non-overlap", window_ms))
    if mode in ("both", "overlap"):
        modes.append(("overlap", overlap_step_ms))

    print(f"EDF: {edf}")
    print(f"EEG: {eeg.shape[1]} channels, {sfreq:g} Hz")
    print(f"Trials: {len(train_trials)} train, {len(test_trials)} test")
    print(f"Window: {window_ms} ms; same model in every test mode")

    results = {}
    csv_rows = []
    for mode_name, step_ms in modes:
        batches, infos = make_batches(
            eeg, timestamps, sfreq, trials, bounds, test_trials,
            window_ms, step_ms,
        )
        features = get_features(batches)
        predictions = model.predict(features)
        results[mode_name] = report_predictions(
            mode_name, trials, test_trials, infos, predictions, window_ms
        )
        if csv_path is not None:
            csv_rows.extend(zip([mode_name] * len(infos), infos, predictions))
        del batches, features

    print("\nComparison (raw predictions, no post-processing):")
    print(f"{'Mode':<16} {'Step':>7} {'Labeled':>8} {'Classification':>15} {'Recognition':>13} {'False gesture':>14} {'Flips/s':>9}")
    for mode_name, step_ms in modes:
        result = results[mode_name]
        print(
            f"{mode_name:<16} {step_ms:>5} ms "
            f"{result['labeled_batches']:>8} "
            f"{result['classification']:>14.1%} "
            f"{result['recognition']:>13.1%} "
            f"{result['false_gesture']:>14.1%} "
            f"{result['flips_per_second']:>9.1f}"
        )
    print("Recognition uses the MOVING triggers, not manual EEG onset labels.")
    print("Overlapping batches are correlated; their counts are not independent samples.")

    if csv_path is not None:
        write_predictions(csv_path, csv_rows)
        print(f"Prediction sequence: {csv_path}")

# ================================================================
# 3. Read command-line options and start the comparison
# ================================================================
# Find the EDF file of the chosen person, then evaluate.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1, help="Person number (1-11)")
    parser.add_argument("--window-ms", "--batch-ms", type=int, default=1000)
    parser.add_argument("--overlap-step-ms", type=int, default=100)
    parser.add_argument("--mode", choices=("both", "non-overlap", "overlap"), default="both")
    parser.add_argument("--train-per-code", type=int, default=4)
    parser.add_argument("--csv", type=Path, help="Save test predictions from both modes")
    args = parser.parse_args()
    paths = sorted(DATA.glob(f"*_Subj_{args.subject:02d}_*.edf"))
    if not paths:
        parser.error(f"No EDF file for subject {args.subject} in {DATA}")
    evaluate(
        paths[0], args.window_ms, args.overlap_step_ms, args.mode,
        args.train_per_code, args.csv,
    )


if __name__ == "__main__":
    main()
