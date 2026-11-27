# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from mne.io import BaseRaw
from dataclasses import dataclass
from typing import ClassVar, Literal

from .data_actor import DataActor
from .trial_data import TrialData

PartitionName = Literal["train", "val", "test"]
# One stim channel for every row of the markers: the movement, then the phase
_STIM_CH_NAMES = ["STI 014", "PHASE"]


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPartition(DataActor):
    name: ClassVar[str] = "dataset_partition"
    partition_name: PartitionName
    trials: list[TrialData]

    @property
    def mne_trials(self) -> list[tuple[str, BaseRaw]]:
        """One Raw for every trial (name, Raw).

        The trials are not joined into one Raw, because they do not all have the same electrodes.
        """
        if not self.trials:
            raise ValueError(
                f"partition {self.partition_name!r} has no trials to build a Raw from"
            )

        return [(trial.file_name, _trial_to_raw(trial)) for trial in self.trials]


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _trial_to_raw(trial: TrialData) -> mne.io.RawArray:
    if trial.signal_type is None:
        raise ValueError(
            f"trial {trial.file_name!r} has no signal_type; cannot type its channels"
        )

    # Stack the per-sample markers under the signal as dedicated stim channels.
    stim_names = _STIM_CH_NAMES[:trial.markers.shape[0]]
    data = np.vstack([trial.signal_ch, trial.markers])
    ch_names = [*trial.channel_names, *stim_names]

    # Type every channel as the signal modality, then retag the markers row.
    info = mne.create_info(
        ch_names=ch_names, sfreq=trial.sfreq, ch_types=trial.signal_type
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_channel_types({name: "stim" for name in stim_names})
    return raw
