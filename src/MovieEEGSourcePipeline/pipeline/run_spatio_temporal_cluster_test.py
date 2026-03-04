import os
import gc
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
# from joblib import parallel_backend

import numpy as np
import mne


def _load_npz_array(path):
    with np.load(path) as npz:
        return npz["X"]


def _get_n_jobs():
    value = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("MNE_NUM_JOBS") or "1"
    return max(1, int(value))


def _get_backend():
    return os.environ.get("JOBLIB_BACKEND", "loky")


def _get_time_decim():
    return max(1, int(os.environ.get("CLUSTER_TIME_DECIM", "1")))


if __name__ == "__main__":
    condition_paths = {
        "lin": "data/stcs_aggregated/X_lin.npz",
        "scr": "data/stcs_aggregated/X_scr.npz",
    }

    with np.load("data/stcs_aggregated/meta.npz") as meta:
        times = meta["times"]

    subjects_dir = Path("data")
    fs_subject = "fsaverage"
    fs_src_fname = subjects_dir / fs_subject / "bem" / "fsaverage-ico-4-src.fif"
    src = mne.read_source_spaces(fs_src_fname, verbose=False)

    base = (-0.2, 0.0)
    bmask = (times >= base[0]) & (times < base[1])
    pmask = times >= 0
    n_jobs = _get_n_jobs()
    backend = _get_backend()
    time_decim = _get_time_decim()

    if not np.any(bmask):
        raise ValueError(f"Baseline window {base} does not overlap the available time axis.")
    if not np.any(pmask):
        raise ValueError("No post-stimulus samples found in the available time axis.")

    n_times_post_full = int(np.count_nonzero(pmask))
    n_times_post = len(range(0, n_times_post_full, time_decim))
    adjacency = mne.spatio_temporal_src_adjacency(src, n_times=n_times_post)
    print(f"Running cluster test with backend={backend}, n_jobs={n_jobs}, time_decim={time_decim}")

    for condition, path in condition_paths.items():
        X_cond = _load_npz_array(path)
        base_mag = np.mean(np.abs(X_cond[:, :, bmask]), axis=2)  # (n_subjects, n_vertices)
        post_mag = np.abs(X_cond[:, :, pmask])  # (n_subjects, n_vertices, n_times)
        if time_decim > 1:
            post_mag = post_mag[:, :, ::time_decim]
        delta = post_mag - base_mag[:, :, np.newaxis]

        # MNE expects (observations, time, space) for spatio-temporal clustering.
        delta = np.transpose(delta, (0, 2, 1))

        # with parallel_backend(backend):
        T_obs, clusters, cluster_pvals, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
                delta,
                adjacency=adjacency,
                n_permutations=2000,
                tail=1,
                n_jobs=n_jobs,
                threshold=None,
                out_type="indices",
                verbose=True,
            )

        results_dir = Path(f"results/spatio_temporal_cluster_test_{condition}")
        results_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            results_dir / "cluster_test_results.npz",
            T_obs=T_obs,
            clusters=np.array(clusters, dtype=object),
            cluster_p_values=cluster_pvals,
            H0=H0,
            time_decim=time_decim,
            n_jobs=n_jobs,
            backend=backend,
        )

        del X_cond, base_mag, post_mag, delta, T_obs, clusters, cluster_pvals, H0
        gc.collect()
