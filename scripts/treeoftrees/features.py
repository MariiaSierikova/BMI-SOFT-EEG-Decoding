"""
Computes simple features of the EEG windows: the numbers a decision tree learns from.

Every window (channels x time points) is summarized by 7 numbers per channel, for
example how strong the signal is or how often it changes direction.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scripts.treeoftrees.moving_data import DATA, cut_windows, load_subject


# ================================================================
# 1. Section: Time-domain features
# ================================================================
# Every feature takes windows (windows x channels x time points) and returns one
# number per window and channel (windows x channels)
def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value: the average size of the signal, ignoring the sign."""
    return np.mean(np.abs(x), axis=2)


def std(x: np.ndarray) -> np.ndarray:
    """Standard deviation: how much the signal moves around its average."""
    return np.std(x, axis=2)


def maxav(x: np.ndarray) -> np.ndarray:
    """Maximum absolute value: the highest peak of the signal."""
    return np.max(np.abs(x), axis=2)


def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square: the strength (energy) of the signal."""
    return np.sqrt(np.mean(x**2, axis=2))


def wl(x: np.ndarray) -> np.ndarray:
    """Waveform length: the total distance the signal travels up and down."""
    return np.sum(np.abs(np.diff(x, axis=2)), axis=2)


def ssc(x: np.ndarray) -> np.ndarray:
    """Slope sign changes: how many times the signal changes direction."""
    dx = np.diff(x, axis=2)
    return np.sum((dx[:, :, :-1] * dx[:, :, 1:]) < 0, axis=2)


def log_det(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log detector: a smooth strength measure, less sensitive to single peaks."""
    return np.exp(np.mean(np.log(np.abs(x) + eps), axis=2))


# ================================================================
# 2. Section: Mapped
# ================================================================
FEATURE_FUNCTIONS = [mav, std, maxav, rms, wl, ssc, log_det]


# ================================================================
# 3. Section: FUNCTIONS
# ================================================================
def get_features(windows: np.ndarray) -> np.ndarray:
    """Compute the 7 features of every channel of every window.

    Returns a table with one row per window and one column per feature and
    channel: first the values of the first feature for all channels, then of the
    second feature, and so on.
    """
    columns = [function(windows) for function in FEATURE_FUNCTIONS]
    return np.concatenate(columns, axis=1)


def get_feature_names(channels: list[str]) -> list[str]:
    """Name the columns of get_features, for example "rms_C4"."""
    return [
        f"{function.__name__}_{channel}"
        for function in FEATURE_FUNCTIONS
        for channel in channels
    ]


# ================================================================
# 4. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load one person and cut the windows
    raw = load_subject(sorted(DATA.glob("*.edf"))[0])
    windows, labels = cut_windows(raw)

    # 2. Compute the features
    features = get_features(windows)
    names = get_feature_names(raw.ch_names)
    print("windows:", windows.shape, "-> features:", features.shape)
    print("first columns:", names[:3], "... last:", names[-1])

    # 3. Look at one feature: the strength (rms) of channel C4 in each class
    column = names.index("rms_C4")
    for name in sorted(set(labels)):
        print(f"{name:15} mean rms_C4 = {features[labels == name, column].mean():.2f}")
