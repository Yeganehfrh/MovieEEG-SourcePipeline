from pathlib import Path
import numpy as np
import pandas as pd

sfreq = 512
tmin = -0.2
rows = []
yeo7 = {
    'N1': 'Visual',
    'N2': 'Somatomotor',
    'N3': 'DorsalAttention',
    'N4': 'VentralAttention',
    'N5': 'Limbic',
    'N6': 'Frontoparietal',
    'N7': 'Default',
    'mwall': 'Medial_Wall',
}

hemisferes = ['lh', 'rh']

roi_names = [yeo7[k] + '_' + hemisferes[i] 
           for k in yeo7.keys() 
           for i in range(len(hemisferes))]

windows = {
    "early": (0.05, 0.15),
    "mid":   (0.15, 0.35),
    "late":  (0.35, 0.80),
}

for fname in Path("data/labels/").glob("*_labels.npz"):
    subj, movie, order, _ = fname.stem.split("_", 3)
    subj = int(subj)

    npz = np.load(fname)
    X = npz[npz.files[0]]  # (epochs, rois, times)
    if X.ndim != 3:
        raise ValueError(f"{fname.name}: expected (epochs, rois, times), got {X.shape}")

    n_epochs, n_rois, n_times = X.shape
    times = np.arange(n_times) / sfreq + tmin
    assert np.isclose(times[0], tmin)

    for win, (t0, t1) in windows.items():
        tmask = (times >= t0) & (times < t1)
        if not np.any(tmask):
            raise ValueError(f"{fname.name}: window {win} empty")

        feat = X[:, :, tmask].mean(axis=-1)

        rows.append(pd.DataFrame({
            "subject": subj,
            "movie": movie,
            "condition": order,
            "epoch": np.repeat(np.arange(n_epochs), n_rois),
            "roi": np.tile(np.array(roi_names), n_epochs),
            "window": win,
            "t_start": t0,
            "t_end": t1,
            "amplitude": feat.reshape(-1),
        }))

df = pd.concat(rows, ignore_index=True)
out_base = Path("data/erp_ready_df")

# Compressed CSV for maximum compatibility
df.to_csv(out_base.with_suffix(".csv.gz"), index=False, compression="gzip")

# Parquet for faster analytics (requires pyarrow or fastparquet)
try:
    df.to_parquet(out_base.with_suffix(".parquet"), index=False)
except Exception as e:
    print(f"Parquet write skipped: {e}")
