from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns


def _get_time_decim(results_npz) -> int:
    if "time_decim" not in results_npz.files:
        return 1
    return max(1, int(np.asarray(results_npz["time_decim"]).item()))


def _load_condition_results(results_root: Path, condition: str):
    path = results_root / f"spatio_temporal_cluster_test_{condition}" / "cluster_test_results.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing results for condition '{condition}': {path}")

    with np.load(path, allow_pickle=True) as npz:
        t_obs = npz["T_obs"]
        clusters = npz["clusters"]
        cluster_p = npz["cluster_p_values"]
        time_decim = _get_time_decim(npz)

    return t_obs, clusters, cluster_p, time_decim, path


def _build_vertex_label_map(labels, hemi_vertices):
    mapping = {int(v): [] for v in hemi_vertices}
    for label in labels:
        verts = np.intersect1d(label.vertices, hemi_vertices)
        for v in verts:
            mapping[int(v)].append(label.name)
    return mapping


def _build_atlas_maps(subjects_dir: Path, fs_subject: str, parc: str, vertices):
    labels = mne.read_labels_from_annot(
        subject=fs_subject,
        parc=parc,
        subjects_dir=str(subjects_dir),
        verbose=False,
    )
    lh_vertices = np.asarray(vertices[0], dtype=int)
    rh_vertices = np.asarray(vertices[1], dtype=int)
    lh_labels = [lbl for lbl in labels if lbl.hemi == "lh"]
    rh_labels = [lbl for lbl in labels if lbl.hemi == "rh"]
    return {
        "lh": _build_vertex_label_map(lh_labels, lh_vertices),
        "rh": _build_vertex_label_map(rh_labels, rh_vertices),
    }


def _global_to_hemi_vertex(global_idx: int, n_lh_vertices: int, vertices):
    if global_idx < n_lh_vertices:
        return "lh", int(vertices[0][global_idx])
    return "rh", int(vertices[1][global_idx - n_lh_vertices])


def _labels_for_cluster_vertices(unique_global_vertices, n_lh_vertices: int, vertices, atlas_maps):
    counts = {}
    for gidx in unique_global_vertices:
        hemi, hemi_vertex = _global_to_hemi_vertex(int(gidx), n_lh_vertices, vertices)
        for lbl in atlas_maps[hemi].get(hemi_vertex, []):
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _top_label_string(label_counts: dict[str, int], n_vertices: int, top_k: int = 3):
    if not label_counts or n_vertices == 0:
        return ""
    parts = []
    top = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    for name, count in top:
        pct = 100.0 * count / n_vertices
        parts.append(f"{name} ({count}/{n_vertices}, {pct:.1f}%)")
    return "; ".join(parts)


def _summarize_condition(
    condition: str,
    t_obs: np.ndarray,
    clusters,
    cluster_p_values: np.ndarray,
    times_post: np.ndarray,
    n_lh_vertices: int,
    vertices: np.ndarray,
    atlas_maps: dict,
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig_ids = np.where(cluster_p_values < alpha)[0]
    sig_mask = np.zeros_like(t_obs, dtype=bool)
    sig_mask_pos = np.zeros_like(t_obs, dtype=bool)
    sig_mask_neg = np.zeros_like(t_obs, dtype=bool)
    rows = []

    for rank, cid in enumerate(sig_ids, start=1):
        time_inds, vertex_inds = clusters[cid]
        time_inds = np.asarray(time_inds, dtype=int)
        vertex_inds = np.asarray(vertex_inds, dtype=int)
        if time_inds.size == 0:
            continue

        sig_mask[time_inds, vertex_inds] = True

        uniq_times = np.unique(time_inds)
        uniq_vertices = np.unique(vertex_inds)
        cluster_t = t_obs[time_inds, vertex_inds]
        peak_local = int(np.argmax(np.abs(cluster_t)))
        peak_time_idx = int(time_inds[peak_local])
        peak_vertex = int(vertex_inds[peak_local])
        peak_t = float(cluster_t[peak_local])
        peak_sign = "pos" if peak_t >= 0 else "neg"

        if peak_sign == "pos":
            sig_mask_pos[time_inds, vertex_inds] = True
        else:
            sig_mask_neg[time_inds, vertex_inds] = True

        peak_hemi, peak_vertex_no = _global_to_hemi_vertex(peak_vertex, n_lh_vertices, vertices)
        peak_label_candidates = atlas_maps[peak_hemi].get(peak_vertex_no, [])
        peak_label = peak_label_candidates[0] if peak_label_candidates else ""

        label_counts = _labels_for_cluster_vertices(uniq_vertices, n_lh_vertices, vertices, atlas_maps)
        top_labels = _top_label_string(label_counts, int(uniq_vertices.size))

        rows.append(
            {
                "condition": condition,
                "cluster_id": int(cid),
                "cluster_rank": int(rank),
                "p_value": float(cluster_p_values[cid]),
                "n_points": int(time_inds.size),
                "n_timepoints": int(uniq_times.size),
                "n_vertices": int(uniq_vertices.size),
                "t_start_sec": float(times_post[uniq_times.min()]),
                "t_end_sec": float(times_post[uniq_times.max()]),
                "peak_time_sec": float(times_post[peak_time_idx]),
                "peak_vertex_idx": peak_vertex,
                "peak_vertex_no": peak_vertex_no,
                "peak_hemi": peak_hemi,
                "peak_label": peak_label,
                "peak_t_obs": peak_t,
                "peak_sign": peak_sign,
                "n_lh_vertices": int(np.sum(uniq_vertices < n_lh_vertices)),
                "n_rh_vertices": int(np.sum(uniq_vertices >= n_lh_vertices)),
                "top_labels": top_labels,
            }
        )

    summary = pd.DataFrame(rows)
    return summary, sig_mask, sig_ids, sig_mask_pos, sig_mask_neg


def _plot_pvalues(summary: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(9, 4))
    x = np.arange(len(summary))
    y = -np.log10(summary["p_value"].to_numpy())
    plt.bar(x, y)
    plt.xticks(x, summary["cluster_id"].astype(str), rotation=45, ha="right")
    plt.xlabel("Cluster ID")
    plt.ylabel("-log10(p)")
    plt.title("Significant Cluster p-values")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_time_coverage(
    sig_mask: np.ndarray,
    t_obs: np.ndarray,
    times_post: np.ndarray,
    out_png: Path,
):
    n_sig_vertices = sig_mask.sum(axis=1)
    max_abs_t_sig = np.where(sig_mask, np.abs(t_obs), np.nan)
    with np.errstate(invalid="ignore"):
        max_abs_t_sig = np.nanmax(max_abs_t_sig, axis=1)
    max_abs_t_sig = np.where(np.isfinite(max_abs_t_sig), max_abs_t_sig, 0.0)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(times_post, n_sig_vertices, color="#1b5e20", lw=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Significant vertices", color="#1b5e20")
    ax1.tick_params(axis="y", labelcolor="#1b5e20")

    ax2 = ax1.twinx()
    ax2.plot(times_post, max_abs_t_sig, color="#c62828", lw=1.5, alpha=0.8)
    ax2.set_ylabel("Max |T_obs| within significant mask", color="#c62828")
    ax2.tick_params(axis="y", labelcolor="#c62828")

    ax1.set_title("Temporal Coverage of Significant Clusters")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_top_vertices_heatmap(
    t_obs: np.ndarray,
    sig_mask: np.ndarray,
    times_post: np.ndarray,
    out_png: Path,
    top_k: int = 64,
):
    sig_per_vertex = sig_mask.sum(axis=0)
    top_vertices = np.where(sig_per_vertex > 0)[0]
    if top_vertices.size == 0:
        return

    top_vertices = top_vertices[np.argsort(sig_per_vertex[top_vertices])[::-1]][:top_k]
    hm = t_obs[:, top_vertices].T

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        hm,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "T_obs"},
        xticklabels=False,
        yticklabels=[str(v) for v in top_vertices],
    )
    tick_positions = np.linspace(0, len(times_post) - 1, 6, dtype=int)
    plt.xticks(tick_positions + 0.5, [f"{times_post[i]:.2f}" for i in tick_positions], rotation=0)
    plt.xlabel("Time (s)")
    plt.ylabel("Vertex index")
    plt.title("T_obs for Top Significant Vertices")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _save_binned_brain_maps(
    sig_mask: np.ndarray,
    times_post: np.ndarray,
    vertices: np.ndarray,
    fs_subject: str,
    subjects_dir: Path,
    out_dir: Path,
    bin_ms: int,
    sign_name: str,
):
    if not np.any(sig_mask):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_sec = bin_ms / 1000.0
    t_min = float(times_post.min())
    t_max = float(times_post.max())
    edges = np.arange(t_min, t_max + bin_sec + 1e-12, bin_sec)

    lh_vertices = np.asarray(vertices[0], dtype=int)
    rh_vertices = np.asarray(vertices[1], dtype=int)

    for b0, b1 in zip(edges[:-1], edges[1:]):
        tmask = (times_post >= b0) & (times_post < b1)
        if not np.any(tmask):
            continue

        occ = sig_mask[tmask].mean(axis=0)
        if np.max(occ) <= 0:
            continue

        stc = mne.SourceEstimate(
            data=occ[:, np.newaxis],
            vertices=[lh_vertices, rh_vertices],
            tmin=0.0,
            tstep=1.0,
            subject=fs_subject,
        )

        vmin = float(np.percentile(occ[occ > 0], 5))
        vmax = float(np.percentile(occ[occ > 0], 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
            vmin, vmax = 0.05, 1.0
        if vmax <= vmin:
            vmax = min(1.0, vmin + 0.05)

        brain = stc.plot(
            subject=fs_subject,
            subjects_dir=str(subjects_dir),
            hemi="split",
            views=["lat", "med"],
            surface="inflated",
            colormap="Reds" if sign_name == "pos" else "Blues",
            background="white",
            colorbar=True,
            time_viewer=False,
            clim=dict(kind="value", lims=[vmin, 0.5 * (vmin + vmax), vmax]),
            size=(1600, 900),
            smoothing_steps=5,
        )

        tag = f"{int(round(b0 * 1000)):04d}_{int(round(b1 * 1000)):04d}ms"
        out_png = out_dir / f"{sign_name}_sig_vertices_{tag}.png"
        brain.save_image(str(out_png))
        brain.close()


def main():
    parser = argparse.ArgumentParser(description="Report significant spatio-temporal clusters.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--meta-path", type=Path, default=Path("data/stcs_aggregated/meta.npz"))
    parser.add_argument("--conditions", nargs="+", default=["lin", "scr"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("results/spatio_temporal_cluster_reports"))
    parser.add_argument("--subjects-dir", type=Path, default=Path("data"))
    parser.add_argument("--fs-subject", type=str, default="fsaverage")
    parser.add_argument("--parc", type=str, default="aparc")
    parser.add_argument("--bin-ms", type=int, default=50)
    parser.add_argument("--make-brain-plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.meta_path) as meta:
        times = meta["times"]
        vertices = meta["vertices"]
        n_lh_vertices = int(len(vertices[0]))

    atlas_maps = _build_atlas_maps(
        subjects_dir=args.subjects_dir,
        fs_subject=args.fs_subject,
        parc=args.parc,
        vertices=vertices,
    )

    all_rows = []
    for condition in args.conditions:
        t_obs, clusters, cluster_pvals, time_decim, source_path = _load_condition_results(args.results_root, condition)
        times_post = times[times >= 0][::time_decim]

        if t_obs.shape[0] != len(times_post):
            raise ValueError(
                f"Time mismatch for condition '{condition}': "
                f"T_obs has {t_obs.shape[0]} rows, but derived times_post has {len(times_post)}."
            )

        summary, sig_mask, sig_ids, sig_mask_pos, sig_mask_neg = _summarize_condition(
            condition=condition,
            t_obs=t_obs,
            clusters=clusters,
            cluster_p_values=cluster_pvals,
            times_post=times_post,
            n_lh_vertices=n_lh_vertices,
            vertices=vertices,
            atlas_maps=atlas_maps,
            alpha=args.alpha,
        )

        cond_out = args.out_dir / condition
        cond_out.mkdir(parents=True, exist_ok=True)
        summary_csv = cond_out / "significant_clusters_summary.csv"
        summary.to_csv(summary_csv, index=False)

        print(
            f"[{condition}] loaded {source_path} | "
            f"total_clusters={len(cluster_pvals)} | significant={len(sig_ids)} | "
            f"summary={summary_csv}"
        )

        if not summary.empty:
            _plot_pvalues(summary, cond_out / "significant_cluster_pvalues.png")
            _plot_time_coverage(sig_mask, t_obs, times_post, cond_out / "significant_time_coverage.png")
            _plot_top_vertices_heatmap(
                t_obs=t_obs,
                sig_mask=sig_mask,
                times_post=times_post,
                out_png=cond_out / "top_significant_vertices_heatmap.png",
            )
            if args.make_brain_plots:
                try:
                    _save_binned_brain_maps(
                        sig_mask=sig_mask_pos,
                        times_post=times_post,
                        vertices=vertices,
                        fs_subject=args.fs_subject,
                        subjects_dir=args.subjects_dir,
                        out_dir=cond_out / "brain_bins",
                        bin_ms=args.bin_ms,
                        sign_name="pos",
                    )
                    _save_binned_brain_maps(
                        sig_mask=sig_mask_neg,
                        times_post=times_post,
                        vertices=vertices,
                        fs_subject=args.fs_subject,
                        subjects_dir=args.subjects_dir,
                        out_dir=cond_out / "brain_bins",
                        bin_ms=args.bin_ms,
                        sign_name="neg",
                    )
                except Exception as exc:
                    print(f"[{condition}] brain plotting skipped due to error: {exc}")

        all_rows.append(summary)

    all_summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_summary_csv = args.out_dir / "significant_clusters_all_conditions.csv"
    all_summary.to_csv(all_summary_csv, index=False)
    print(f"[done] combined summary: {all_summary_csv}")


if __name__ == "__main__":
    main()
