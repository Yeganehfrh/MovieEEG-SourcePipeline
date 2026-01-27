from pathlib import Path

import numpy as np
import mne

# Forward / Inverse builders
# -----------------------------
def _load_epochs(fname: Path, sfreq=512) -> mne.Epochs:
    epochs = mne.read_epochs(fname, preload=True, verbose=False)

    # Resample only if needed (to increase the computation speed)
    if epochs.info["sfreq"] != sfreq:
        epochs.resample(sfreq, npad="auto", verbose=False)

    epochs.pick('eeg')

    return epochs


def make_forward(example_epochs_fpath: Path, FS_SUBJECT, FS_SRC_FNAME, FS_BEM_FNAME) -> mne.Forward:
    """
    Build a single fsaverage forward solution to reuse across all subjects.
    Uses an example epochs file to define channel set and measurement info.
    """
    epochs = _load_epochs(example_epochs_fpath)

    # fsaverage transform (built-in)
    trans = FS_SUBJECT
    src = mne.read_source_spaces(FS_SRC_FNAME, verbose=False)
    bem = mne.read_bem_solution(FS_BEM_FNAME, verbose=False)

    # Forward solution
    fwd = mne.make_forward_solution(
        info=epochs.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        verbose=False,
    )
    return fwd


def make_inverse_from_baseline(
    baseline_epochs: mne.Epochs,
    fwd: mne.Forward,
) -> mne.minimum_norm.InverseOperator:
    """
    Build an inverse operator using baseline epochs to estimate noise covariance.
    Critically: covariance is computed directly from epochs (no pseudo-continuous stitching).
    """

    # Noise covariance from baseline epochs
    cov = mne.compute_covariance(
        baseline_epochs,
        method="shrunk", # 'shrunk' is stable for EEG; rank='info' respects projections
        rank="info",
        verbose=False,
    )

    inv = mne.minimum_norm.make_inverse_operator(
        info=baseline_epochs.info,
        forward=fwd,
        noise_cov=cov,
        loose=0.2,   # common for cortical source models
        depth=0.8,   # typical depth weighting
        verbose=False,
    )
    return inv

# Source -> labels
# -----------------------------
def extract_label_time_series(
    epochs: mne.Epochs,
    inv: mne.minimum_norm.InverseOperator,
    atlas_labels: list,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Apply inverse on cut-locked epochs and extract parcel time courses.
    Returns: array of shape (n_epochs, n_labels, n_times)
    """
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator=inv,
        method="eLORETA",
        lambda2=1.0 / 9.0,
        pick_ori="normal",     # explicit orientation choice
        return_generator=False,
        verbose=False,
    )

    label_ts = mne.extract_label_time_course(
        stcs,
        labels=atlas_labels,
        src=inv["src"],
        mode="pca_flip",       # avoids sign cancellation; good for connectivity/ERP
        return_generator=False,
        verbose=False,
    )
    # label_ts: (n_epochs, n_labels, n_times)
    return label_ts
