from pathlib import Path
import joblib
import time
import os
import re
import numpy as np
import mne
from src.MovieEEGSourcePipeline.source import _load_epochs, make_forward, make_inverse_from_baseline, extract_label_time_series


def run_source_localisation(subject_dir, data_dir, fs_subject, fs_src_fname, fs_bem_fname, n_jobs=None):
    # Atlas labels (fsaverage annotations)
    atlas_labels = mne.read_labels_from_annot(
        subject=fs_subject,
        parc="Yeo2011_7Networks_N1000",
        subjects_dir=str(subject_dir),
        verbose=False,
    )

    # Pick a single example file to define channel set for forward model
    example = next(data_dir.glob("*_city_l_epo.fif"))
    fwd = make_forward(example, FS_SUBJECT=fs_subject, FS_SRC_FNAME=fs_src_fname, FS_BEM_FNAME=fs_bem_fname)

    inv_cache = {}  # subject -> inverse operator built from baseline1

    for epochs_path in sorted(data_dir.glob("*_epo.fif")):
        m = re.search(r"^(\d+)_([^_]+_[^_]+)_epo$", epochs_path.stem)
        if m is None:
            continue
        subject, film = m.groups()
        epochs = _load_epochs(epochs_path)

        out_dir = Path("data/labels")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path_labels = out_dir / f"{subject}_{film}_labels.npz"

        # if output_path_labels.exists():
        #     continue

        # Ensure we have an inverse per subject (from baseline1)
        if subject not in inv_cache:
            # pick baseline portion (and avoid immediate pre-cut because of anticipatory activity)
            epochs_base = epochs.copy().crop(tmin=-0.2, tmax=-0.05)
            inv_cache[subject] = make_inverse_from_baseline(epochs_base, fwd)

        inv = inv_cache[subject]

        print(f">>>>>>>> {subject} {film}")
        label_ts = extract_label_time_series(epochs, inv, atlas_labels, n_jobs=n_jobs)
        np.savez_compressed(output_path_labels, labels=label_ts)


if __name__ == '__main__':
    # Config
    DATA_DIR = Path("data/epochs")
    SUBJECTS_DIR = Path("data")          # contains fsaverage/ (i.e., data/fsaverage)
    FS_SUBJECT = "fsaverage"             # we use fsaverage anatomy for everyone

    # Use an ico-4 source space (coarse; appropriate for Yeo-7 parcellation)
    FS_SRC_FNAME = SUBJECTS_DIR / FS_SUBJECT / "bem" / "fsaverage-ico-4-src.fif"
    FS_BEM_FNAME = SUBJECTS_DIR / FS_SUBJECT / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    os.environ["SUBJECTS_DIR"] = str(SUBJECTS_DIR)  # a temporary solution to make sure mne can find SUBJECTS_DIR

    n_jobs = int(os.environ.get("MNE_NUM_JOBS", "-1"))
    print(f'core numbers are {joblib.effective_n_jobs(n_jobs)}')

    start = time.time()
    print(f'Start time: {start}')

    run_source_localisation(
        subject_dir=SUBJECTS_DIR,
        data_dir=DATA_DIR,
        fs_subject=FS_SUBJECT,
        fs_src_fname=FS_SRC_FNAME,
        fs_bem_fname=FS_BEM_FNAME,
        n_jobs=n_jobs,
    )
    
    end = time.time()
    print(f'Elapsed: {end - start:.2f} seconds')
