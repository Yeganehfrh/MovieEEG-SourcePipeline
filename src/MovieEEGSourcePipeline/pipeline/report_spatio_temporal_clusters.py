from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _get_time_decim(results_npz) -> int:
    if "time_decim" not in results_npz.files:
        return 1
    return max(1, int(np.asarray(results_npz["time_decim"]).item()))


def _load_condition_results(results_root: Path, condition: str):
    path = results_root / f"spatio_temporal_cluster_test_{condition}" / "cluster_test_results_signed_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing results for condition '{condition}': {path}")

    with np.load(path, allow_pickle=True) as npz:
        t_obs = npz["T_obs"]
        clusters = npz["clusters"]
        cluster_p = npz["cluster_p_values"]
        time_decim = _get_time_decim(npz)

    return t_obs, clusters, cluster_p, time_decim, path


def _summarize_condition(
    condition: str,
    t_obs: np.ndarray,
    clusters,
    cluster_p_values: np.ndarray,
    times_post: np.ndarray,
    n_lh_vertices: int,
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sig_ids = np.where(cluster_p_values < alpha)[0]
    sig_mask = np.zeros_like(t_obs, dtype=bool)
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
        peak_local = int(np.argmax(cluster_t))
        peak_time_idx = int(time_inds[peak_local])
        peak_vertex = int(vertex_inds[peak_local])

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
                "peak_t_obs": float(cluster_t[peak_local]),
                "n_lh_vertices": int(np.sum(uniq_vertices < n_lh_vertices)),
                "n_rh_vertices": int(np.sum(uniq_vertices >= n_lh_vertices)),
            }
        )

    summary = pd.DataFrame(rows)
    return summary, sig_mask, sig_ids


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
    max_t_sig = np.where(sig_mask, t_obs, np.nan)
    with np.errstate(invalid="ignore"):
        max_t_sig = np.nanmax(max_t_sig, axis=1)
    max_t_sig = np.where(np.isfinite(max_t_sig), max_t_sig, 0.0)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(times_post, n_sig_vertices, color="#1b5e20", lw=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Significant vertices", color="#1b5e20")
    ax1.tick_params(axis="y", labelcolor="#1b5e20")

    ax2 = ax1.twinx()
    ax2.plot(times_post, max_t_sig, color="#c62828", lw=1.5, alpha=0.8)
    ax2.set_ylabel("Max T_obs within significant mask", color="#c62828")
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


def main():
    parser = argparse.ArgumentParser(description="Report significant spatio-temporal clusters.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--meta-path", type=Path, default=Path("data/stcs_aggregated/meta.npz"))
    parser.add_argument("--conditions", nargs="+", default=["lin", "scr"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=Path("results/spatio_temporal_cluster_reports_signed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.meta_path) as meta:
        times = meta["times"]
        vertices = meta["vertices"]
        n_lh_vertices = int(len(vertices[0]))

    all_rows = []
    for condition in args.conditions:
        t_obs, clusters, cluster_pvals, time_decim, source_path = _load_condition_results(args.results_root, condition)
        times_post = times[times >= 0][::time_decim]

        if t_obs.shape[0] != len(times_post):
            raise ValueError(
                f"Time mismatch for condition '{condition}': "
                f"T_obs has {t_obs.shape[0]} rows, but derived times_post has {len(times_post)}."
            )

        summary, sig_mask, sig_ids = _summarize_condition(
            condition=condition,
            t_obs=t_obs,
            clusters=clusters,
            cluster_p_values=cluster_pvals,
            times_post=times_post,
            n_lh_vertices=n_lh_vertices,
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

        all_rows.append(summary)

    all_summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_summary_csv = args.out_dir / "significant_clusters_all_conditions.csv"
    all_summary.to_csv(all_summary_csv, index=False)
    print(f"[done] combined summary: {all_summary_csv}")


if __name__ == "__main__":
    main()
