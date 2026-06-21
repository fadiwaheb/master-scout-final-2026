"""
clustering.py — Stage 10.  USE #1 (clustering).
Group players into PLAY STYLES with K-Means (e.g. pacey forwards, technical
midfielders, physical center-backs).

Functions:
    run_player_kmeans(df, n_clusters)      -> df with cluster_id + fitted model
    elbow_analysis(df, k_range)            -> inertia/silhouette per k (+ chart)
    describe_cluster(df, cluster_id)       -> verbal description + sample players
    get_players_from_cluster(df, cluster_id, **filters)

k is chosen with the ELBOW method (chart saved to reports/figures/). Clustering
runs on outfield players with complete attribute features; GKs / players without
features get cluster_id = -1. The cluster_id is written back into
final_scouting_table.csv.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # backend left as-is: savefig works headless,
                                  # and notebooks keep their inline backend
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import SIM_FEATURES   # reuse the same play-style feature set

# Chosen number of clusters. Reasoning documented in the decisions doc and set
# from the Elbow chart (the bend) + interpretability. Override via run_player_kmeans.
DEFAULT_K = 5
RANDOM_STATE = 42

# the 6 face stats used to label clusters verbally
FACE = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]


def _feature_matrix(df):
    """Return (outfield_df_with_features, scaled_X, scaler)."""
    pool = df[df["position_group"] != "GK"].dropna(subset=SIM_FEATURES).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(pool[SIM_FEATURES].astype(float))
    return pool, X, scaler


def elbow_analysis(df, k_range=range(2, 11), save_path=None):
    """Compute inertia + silhouette for each k; optionally save the Elbow chart."""
    pool, X, _ = _feature_matrix(df)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    res = pd.DataFrame(rows)

    if save_path is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(res["k"], res["inertia"], "o-", color="#264653", label="Inertia")
        ax1.set_xlabel("k (number of clusters)")
        ax1.set_ylabel("Inertia (within-cluster SS)", color="#264653")
        ax2 = ax1.twinx()
        ax2.plot(res["k"], res["silhouette"], "s--", color="#e76f51", label="Silhouette")
        ax2.set_ylabel("Silhouette score", color="#e76f51")
        plt.title("K-Means — Elbow method (inertia) + Silhouette")
        fig.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
    return res


def run_player_kmeans(df, n_clusters=DEFAULT_K):
    """Fit K-Means on play-style features. Returns (df_with_cluster_id, model, scaler).
    Outfield players get a cluster 0..k-1; everyone else gets -1."""
    pool, X, scaler = _feature_matrix(df)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X)

    out = df.copy()
    out["cluster_id"] = -1
    out.loc[pool.index, "cluster_id"] = km.labels_
    return out, km, scaler


def best_k(df, k_range=range(3, 7)):
    """Let the ALGORITHM choose K: the value in k_range (3-6) with the highest
    silhouette score. Returns (best_k, best_silhouette)."""
    pool, X, _ = _feature_matrix(df)
    best, best_s = DEFAULT_K, -1.0
    for k in k_range:
        if len(pool) <= k:
            continue
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_s:
            best, best_s = k, s
    return best, round(float(best_s), 3)


def cluster_xy(labeled, max_points=900):
    """2D PCA projection of the clustered players (sampled) for a scatter plot.
    Returns (xy ndarray [n,2], cluster_ids ndarray [n])."""
    pool = labeled[labeled["cluster_id"] != -1].dropna(subset=SIM_FEATURES)
    if len(pool) > max_points:
        per = max(1, max_points // max(1, pool["cluster_id"].nunique()))
        parts = [g.sample(min(len(g), per), random_state=RANDOM_STATE)
                 for _, g in pool.groupby("cluster_id")]
        pool = pd.concat(parts)
    X = StandardScaler().fit_transform(pool[SIM_FEATURES].astype(float))
    xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    return xy, pool["cluster_id"].to_numpy()


TRAIT_WORDS = {
    "pace": "fast", "shooting": "goal-scoring", "passing": "playmaking",
    "dribbling": "technical", "defending": "defensive", "physic": "physical",
}


def _label_from_traits(traits):
    """Heuristic verbal label from the dominant attributes."""
    return " / ".join(TRAIT_WORDS.get(t, t) for t in traits)


def describe_cluster(df, cluster_id):
    """Verbal description of a cluster: size, a generated label, the 5 most
    dominant attributes (mean value + how far above the average they sit — the
    NUMERIC justification for the label), and a few representative players."""
    members = df[df["cluster_id"] == cluster_id]
    means = members[FACE].mean().round(1)

    # rank the face stats by their MEAN VALUE — the strongest attributes of this
    # style. The label and the numbers then agree (and explain each other).
    ranked = list(means.sort_values(ascending=False).index)

    # the 5 dominant (strongest) traits with their mean value — numeric evidence
    dominant = [{"trait": t, "mean": float(means[t])} for t in ranked[:5]]

    sample = members.sort_values("overall", ascending=False)["short_name"].head(5).tolist()
    pos_mix = members["position_group"].value_counts(normalize=True).round(2).to_dict()
    return {
        "cluster_id": cluster_id,
        "label": _label_from_traits(ranked[:2]),
        "ranked": ranked,
        "size": len(members),
        "mean_face_stats": means.to_dict(),
        "dominant_traits": dominant,
        "position_mix": pos_mix,
        "sample_players": sample,
    }


def describe_clusters(df, ids):
    """Describe every cluster AND guarantee each gets a DISTINCT verbal label
    (two clusters never share the same characterization)."""
    descs = [describe_cluster(df, c) for c in ids]
    used = set()
    for d in descs:
        ranked = d["ranked"]
        label = _label_from_traits(ranked[:2])
        k = 2  # if the label is taken, swap the 2nd trait down the ranking
        while label in used and k < len(ranked):
            label = _label_from_traits([ranked[0], ranked[k]])
            k += 1
        if label in used and len(ranked) >= 3:  # last resort: add a 3rd trait
            label = label + " / " + TRAIT_WORDS.get(ranked[2], ranked[2])
        used.add(label)
        d["label"] = label
    return descs


def plot_clusters_2d(df, save_path=None, ax=None):
    """Project the play-style features to 2D with PCA and scatter players colored
    by cluster_id, with a legend of the verbal cluster labels. A clear visual of
    how the styles separate. Requires df to already have a cluster_id column."""
    pool = df[(df["cluster_id"] != -1)].dropna(subset=SIM_FEATURES).copy()
    X = StandardScaler().fit_transform(pool[SIM_FEATURES].astype(float))
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    pool["pc1"], pool["pc2"] = coords[:, 0], coords[:, 1]

    labels = {c: describe_cluster(df, c)["label"] for c in sorted(pool["cluster_id"].unique())}
    palette = plt.get_cmap("tab10")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    for c in sorted(pool["cluster_id"].unique()):
        sub = pool[pool["cluster_id"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.45,
                   color=palette(c % 10), label=f"{c}: {labels[c]}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("K-Means play-style clusters (PCA 2D projection)")
    ax.legend(title="cluster", markerscale=2, fontsize=9, loc="best")
    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=120)
        plt.close()
    return labels


def get_players_from_cluster(df, cluster_id, position_group=None, max_age=None,
                             max_value_eur=None, top_n=20):
    """Return players from a cluster, optionally filtered, sorted by overall."""
    res = df[df["cluster_id"] == cluster_id]
    if position_group is not None:
        res = res[res["position_group"] == position_group]
    if max_age is not None:
        res = res[res["age"] <= max_age]
    if max_value_eur is not None:
        res = res[res["value_eur"] <= max_value_eur]
    cols = ["short_name", "position_group", "age", "overall", "value_eur"]
    return res.sort_values("overall", ascending=False).head(top_n)[cols].reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports/figures"
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    print("=" * 70, "\nSTAGE 10 — K-Means play-style clustering\n", "=" * 70)

    # 1) Elbow analysis
    elbow = elbow_analysis(df, save_path=fig_dir / "10_kmeans_elbow.png")
    print("\nElbow / silhouette by k:")
    print(elbow.to_string(index=False))
    print(f"\nChosen k = {DEFAULT_K} (elbow bend + interpretable styles). "
          f"Chart -> {fig_dir/'10_kmeans_elbow.png'}")

    # 2) Fit and label
    clustered, km, scaler = run_player_kmeans(df, n_clusters=DEFAULT_K)
    out = root / "data/processed/final_scouting_table.csv"
    clustered.to_csv(out, index=False)
    print(f"\ncluster_id written back to {out.name}")

    # quality checks
    n_out = int((clustered["cluster_id"] != -1).sum())
    print(f"clustered outfield players: {n_out:,}; unclustered (GK/no-features): "
          f"{int((clustered['cluster_id'] == -1).sum()):,}")

    # 3) Describe each cluster
    print("\n" + "=" * 70, "\nCLUSTER DESCRIPTIONS\n", "=" * 70)
    for cid in range(DEFAULT_K):
        d = describe_cluster(clustered, cid)
        print(f"\n[Cluster {cid}] — \"{d['label']}\"  (n={d['size']:,})")
        print("  mean face stats:", d["mean_face_stats"])
        print("  position mix:", d["position_mix"])
        print("  sample:", ", ".join(d["sample_players"]))

    # 4) Colored 2D scatter of the clusters (PCA projection)
    plot_clusters_2d(clustered, save_path=fig_dir / "10_kmeans_clusters.png")
    print(f"\nCluster scatter -> {fig_dir/'10_kmeans_clusters.png'}")


if __name__ == "__main__":
    main()
