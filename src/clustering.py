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

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא sys כדי להוסיף את src לנתיב הייבוא
import sys

# ייבוא numpy לחישובים מספריים
import numpy as np
# ייבוא pandas לעבודה עם הטבלה
import pandas as pd
# ייבוא matplotlib לציור גרפים (השומר עובד גם ללא תצוגה)
import matplotlib.pyplot as plt   # backend left as-is: savefig works headless,
                                  # and notebooks keep their inline backend
# ייבוא StandardScaler לנרמול z-score
from sklearn.preprocessing import StandardScaler
# ייבוא אלגוריתם K-Means
from sklearn.cluster import KMeans
# ייבוא PCA להטלה דו-ממדית לתצוגה
from sklearn.decomposition import PCA
# ייבוא silhouette_score להערכת איכות הקיבוץ
from sklearn.metrics import silhouette_score

# מוסיפים את תיקיית הקובץ לנתיב כדי לייבא מודול אח
sys.path.insert(0, str(Path(__file__).resolve().parent))
# משתמשים באותו סט תכונות סגנון-משחק כמו במודול הדמיון
from similarity import SIM_FEATURES   # reuse the same play-style feature set

# Chosen number of clusters. Reasoning documented in the decisions doc and set
# from the Elbow chart (the bend) + interpretability. Override via run_player_kmeans.
# מספר הקלאסטרים בברירת מחדל (נבחר מגרף ה-Elbow + פרשנות)
DEFAULT_K = 5
# זרע אקראיות קבוע לשחזוריות
RANDOM_STATE = 42

# the 6 face stats used to label clusters verbally
# 6 תכונות הפנים ששמשות לתיוג מילולי של הקלאסטרים
FACE = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]


# פונקציית עזר: מחזירה (מאגר שחקני שדה עם תכונות, מטריצה מנורמלת, ה-scaler)
def _feature_matrix(df):
    """Return (outfield_df_with_features, scaled_X, scaler)."""
    # מאגר: שחקני שדה בלבד עם כל תכונות הסגנון
    pool = df[df["position_group"] != "GK"].dropna(subset=SIM_FEATURES).copy()
    # מנרמל z-score
    scaler = StandardScaler()
    # מטריצת התכונות המנורמלת
    X = scaler.fit_transform(pool[SIM_FEATURES].astype(float))
    # מחזירים את שלושתם
    return pool, X, scaler


# ניתוח Elbow: אינרציה וסילואט לכל k, ושמירת גרף אופציונלית
def elbow_analysis(df, k_range=range(2, 11), save_path=None):
    """Compute inertia + silhouette for each k; optionally save the Elbow chart."""
    # מכינים את מטריצת התכונות
    pool, X, _ = _feature_matrix(df)
    # רשימה לאיסוף תוצאות לכל k
    rows = []
    # עוברים על כל ערך k בטווח
    for k in k_range:
        # מאמנים K-Means עם k קלאסטרים
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        # שומרים את האינרציה ואת ציון הסילואט
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    # ממירים לטבלה
    res = pd.DataFrame(rows)

    # אם התבקש נתיב שמירה — מציירים את גרף ה-Elbow
    if save_path is not None:
        # יוצרים ציר ראשי לאינרציה
        fig, ax1 = plt.subplots(figsize=(8, 5))
        # עקומת האינרציה
        ax1.plot(res["k"], res["inertia"], "o-", color="#264653", label="Inertia")
        # תווית ציר X
        ax1.set_xlabel("k (number of clusters)")
        # תווית ציר Y הראשי
        ax1.set_ylabel("Inertia (within-cluster SS)", color="#264653")
        # ציר משני לסילואט
        ax2 = ax1.twinx()
        # עקומת הסילואט
        ax2.plot(res["k"], res["silhouette"], "s--", color="#e76f51", label="Silhouette")
        # תווית ציר Y המשני
        ax2.set_ylabel("Silhouette score", color="#e76f51")
        # כותרת הגרף
        plt.title("K-Means — Elbow method (inertia) + Silhouette")
        # פריסה מהודקת
        fig.tight_layout()
        # שמירת הגרף לקובץ
        plt.savefig(save_path, dpi=120)
        # סגירת הדמות לשחרור זיכרון
        plt.close()
    # מחזירים את טבלת התוצאות
    return res


# מאמן K-Means על תכונות הסגנון ומחזיר (טבלה עם cluster_id, מודל, scaler)
def run_player_kmeans(df, n_clusters=DEFAULT_K):
    """Fit K-Means on play-style features. Returns (df_with_cluster_id, model, scaler).
    Outfield players get a cluster 0..k-1; everyone else gets -1."""
    # מכינים את מטריצת התכונות
    pool, X, scaler = _feature_matrix(df)
    # מאמנים את המודל
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X)

    # עותק של הטבלה לכתיבת התוצאה
    out = df.copy()
    # ברירת מחדל: כל השחקנים מקבלים -1 (לא מקובצים)
    out["cluster_id"] = -1
    # שחקני השדה שקובצו מקבלים את התווית שלהם
    out.loc[pool.index, "cluster_id"] = km.labels_
    # מחזירים את הטבלה, המודל וה-scaler
    return out, km, scaler


# בוחר את K האופטימלי: הערך בטווח עם ציון הסילואט הגבוה ביותר
def best_k(df, k_range=range(3, 7)):
    """Let the ALGORITHM choose K: the value in k_range (3-6) with the highest
    silhouette score. Returns (best_k, best_silhouette)."""
    # מכינים את מטריצת התכונות
    pool, X, _ = _feature_matrix(df)
    # אתחול הטוב ביותר וערך הסילואט הטוב ביותר
    best, best_s = DEFAULT_K, -1.0
    # עוברים על כל k
    for k in k_range:
        # מדלגים אם אין מספיק נקודות
        if len(pool) <= k:
            continue
        # מקבצים ומקבלים תוויות
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        # מחשבים ציון סילואט
        s = silhouette_score(X, labels)
        # אם השתפר — מעדכנים את הטוב ביותר
        if s > best_s:
            best, best_s = k, s
    # מחזירים את ה-K הטוב ביותר ואת הסילואט
    return best, round(float(best_s), 3)


# מחזיר הטלת PCA דו-ממדית של השחקנים המקובצים (לדגימה) עבור פיזור
def cluster_xy(labeled, max_points=900):
    """2D PCA projection of the clustered players (sampled) for a scatter plot.
    Returns (xy ndarray [n,2], cluster_ids ndarray [n])."""
    # רק שחקנים מקובצים עם תכונות מלאות
    pool = labeled[labeled["cluster_id"] != -1].dropna(subset=SIM_FEATURES)
    # אם יש יותר מדי נקודות — דוגמים באופן מאוזן בין הקלאסטרים
    if len(pool) > max_points:
        # כמה נקודות לכל קלאסטר
        per = max(1, max_points // max(1, pool["cluster_id"].nunique()))
        # דוגמים מכל קבוצה
        parts = [g.sample(min(len(g), per), random_state=RANDOM_STATE)
                 for _, g in pool.groupby("cluster_id")]
        # מאחדים בחזרה
        pool = pd.concat(parts)
    # מנרמלים את התכונות
    X = StandardScaler().fit_transform(pool[SIM_FEATURES].astype(float))
    # מטילים ל-2 ממדים עם PCA
    xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    # מחזירים את הקואורדינטות ואת תוויות הקלאסטר
    return xy, pool["cluster_id"].to_numpy()


# מילון: תכונה דומיננטית → מילת תיאור סגנון
TRAIT_WORDS = {
    "pace": "fast", "shooting": "goal-scoring", "passing": "playmaking",
    "dribbling": "technical", "defending": "defensive", "physic": "physical",
}


# פונקציית עזר: בונה תווית מילולית מהתכונות הדומיננטיות
def _label_from_traits(traits):
    """Heuristic verbal label from the dominant attributes."""
    # מחברים את מילות התיאור של התכונות
    return " / ".join(TRAIT_WORDS.get(t, t) for t in traits)


# מתאר קלאסטר אחד: גודל, תווית, 5 התכונות הדומיננטיות ושחקנים מייצגים
def describe_cluster(df, cluster_id):
    """Verbal description of a cluster: size, a generated label, the 5 most
    dominant attributes (mean value + how far above the average they sit — the
    NUMERIC justification for the label), and a few representative players."""
    # חברי הקלאסטר
    members = df[df["cluster_id"] == cluster_id]
    # ממוצע 6 תכונות הפנים בקלאסטר
    means = members[FACE].mean().round(1)

    # rank the face stats by their MEAN VALUE — the strongest attributes of this
    # style. The label and the numbers then agree (and explain each other).
    # מדרגים את התכונות לפי הממוצע — מהחזקה לחלשה
    ranked = list(means.sort_values(ascending=False).index)

    # the 5 dominant (strongest) traits with their mean value — numeric evidence
    # 5 התכונות הדומיננטיות עם ערך הממוצע (ההצדקה המספרית לתווית)
    dominant = [{"trait": t, "mean": float(means[t])} for t in ranked[:5]]

    # 5 שחקנים מייצגים (בעלי הדירוג הגבוה ביותר)
    sample = members.sort_values("overall", ascending=False)["short_name"].head(5).tolist()
    # תמהיל העמדות בקלאסטר (באחוזים)
    pos_mix = members["position_group"].value_counts(normalize=True).round(2).to_dict()
    # מחזירים מילון תיאור מלא
    return {
        "cluster_id": cluster_id,
        # תווית מ-2 התכונות הדומיננטיות
        "label": _label_from_traits(ranked[:2]),
        "ranked": ranked,
        "size": len(members),
        "mean_face_stats": means.to_dict(),
        "dominant_traits": dominant,
        "position_mix": pos_mix,
        "sample_players": sample,
    }


# מתאר את כל הקלאסטרים ומבטיח שלכל אחד תהיה תווית ייחודית
def describe_clusters(df, ids):
    """Describe every cluster AND guarantee each gets a DISTINCT verbal label
    (two clusters never share the same characterization)."""
    # מתארים כל קלאסטר
    descs = [describe_cluster(df, c) for c in ids]
    # קבוצת התוויות שכבר נוצלו
    used = set()
    # עוברים על כל תיאור
    for d in descs:
        # דירוג התכונות של הקלאסטר
        ranked = d["ranked"]
        # תווית התחלתית מ-2 התכונות הראשונות
        label = _label_from_traits(ranked[:2])
        # אם התווית תפוסה — מחליפים את התכונה השנייה בהמשך הדירוג
        k = 2  # if the label is taken, swap the 2nd trait down the ranking
        while label in used and k < len(ranked):
            label = _label_from_traits([ranked[0], ranked[k]])
            k += 1
        # מוצא אחרון: מוסיפים תכונה שלישית
        if label in used and len(ranked) >= 3:  # last resort: add a 3rd trait
            label = label + " / " + TRAIT_WORDS.get(ranked[2], ranked[2])
        # מסמנים את התווית כנוצלה
        used.add(label)
        # מעדכנים את התווית בתיאור
        d["label"] = label
    # מחזירים את התיאורים
    return descs


# מצייר את הקלאסטרים ב-2D (הטלת PCA), צבע לכל קלאסטר + מקרא תוויות
def plot_clusters_2d(df, save_path=None, ax=None):
    """Project the play-style features to 2D with PCA and scatter players colored
    by cluster_id, with a legend of the verbal cluster labels. A clear visual of
    how the styles separate. Requires df to already have a cluster_id column."""
    # רק שחקנים מקובצים עם תכונות מלאות
    pool = df[(df["cluster_id"] != -1)].dropna(subset=SIM_FEATURES).copy()
    # נרמול התכונות
    X = StandardScaler().fit_transform(pool[SIM_FEATURES].astype(float))
    # הטלת PCA ל-2 ממדים
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    # מוסיפים עמודות לקואורדינטות
    pool["pc1"], pool["pc2"] = coords[:, 0], coords[:, 1]

    # תוויות מילוליות לכל קלאסטר
    labels = {c: describe_cluster(df, c)["label"] for c in sorted(pool["cluster_id"].unique())}
    # פלטת צבעים
    palette = plt.get_cmap("tab10")

    # האם אנו יוצרים דמות משלנו (או מציירים לתוך ציר נתון)
    own_fig = ax is None
    # אם כן — יוצרים דמות וציר
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    # מציירים פיזור לכל קלאסטר בצבע משלו
    for c in sorted(pool["cluster_id"].unique()):
        # חברי הקלאסטר
        sub = pool[pool["cluster_id"] == c]
        # פיזור הנקודות
        ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.45,
                   color=palette(c % 10), label=f"{c}: {labels[c]}")
    # תוויות הצירים
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    # כותרת
    ax.set_title("K-Means play-style clusters (PCA 2D projection)")
    # מקרא
    ax.legend(title="cluster", markerscale=2, fontsize=9, loc="best")
    # אם יצרנו דמות משלנו — מהדקים, שומרים וסוגרים
    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=120)
        plt.close()
    # מחזירים את מילון התוויות
    return labels


# מחזיר שחקנים מקלאסטר נתון, עם סינון אופציונלי, ממוין לפי דירוג
def get_players_from_cluster(df, cluster_id, position_group=None, max_age=None,
                             max_value_eur=None, top_n=20):
    """Return players from a cluster, optionally filtered, sorted by overall."""
    # חברי הקלאסטר
    res = df[df["cluster_id"] == cluster_id]
    # סינון עמדה אופציונלי
    if position_group is not None:
        res = res[res["position_group"] == position_group]
    # גיל מקסימלי אופציונלי
    if max_age is not None:
        res = res[res["age"] <= max_age]
    # שווי מקסימלי אופציונלי
    if max_value_eur is not None:
        res = res[res["value_eur"] <= max_value_eur]
    # עמודות התצוגה
    cols = ["short_name", "position_group", "age", "overall", "value_eur"]
    # ממיינים לפי דירוג ומחזירים top_n
    return res.sort_values("overall", ascending=False).head(top_n)[cols].reset_index(drop=True)


# בלוק שמורץ בהרצה ישירה — מריץ Elbow, מקבץ, מתאר ומצייר
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # תיקיית הגרפים
    fig_dir = root / "reports/figures"
    # טוענים את הטבלה המרכזית
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    # כותרת
    print("=" * 70, "\nSTAGE 10 — K-Means play-style clustering\n", "=" * 70)

    # 1) Elbow analysis
    # מריצים ניתוח Elbow ושומרים גרף
    elbow = elbow_analysis(df, save_path=fig_dir / "10_kmeans_elbow.png")
    # מדפיסים את הטבלה
    print("\nElbow / silhouette by k:")
    print(elbow.to_string(index=False))
    # מציינים את ה-K שנבחר
    print(f"\nChosen k = {DEFAULT_K} (elbow bend + interpretable styles). "
          f"Chart -> {fig_dir/'10_kmeans_elbow.png'}")

    # 2) Fit and label
    # מאמנים את הקיבוץ
    clustered, km, scaler = run_player_kmeans(df, n_clusters=DEFAULT_K)
    # כותבים את cluster_id חזרה לטבלה המרכזית
    out = root / "data/processed/final_scouting_table.csv"
    clustered.to_csv(out, index=False)
    print(f"\ncluster_id written back to {out.name}")

    # quality checks
    # כמה שחקני שדה קובצו
    n_out = int((clustered["cluster_id"] != -1).sum())
    print(f"clustered outfield players: {n_out:,}; unclustered (GK/no-features): "
          f"{int((clustered['cluster_id'] == -1).sum()):,}")

    # 3) Describe each cluster
    # מתארים כל קלאסטר
    print("\n" + "=" * 70, "\nCLUSTER DESCRIPTIONS\n", "=" * 70)
    for cid in range(DEFAULT_K):
        # תיאור הקלאסטר
        d = describe_cluster(clustered, cid)
        # כותרת הקלאסטר
        print(f"\n[Cluster {cid}] — \"{d['label']}\"  (n={d['size']:,})")
        # ממוצעי תכונות הפנים
        print("  mean face stats:", d["mean_face_stats"])
        # תמהיל העמדות
        print("  position mix:", d["position_mix"])
        # שחקנים מייצגים
        print("  sample:", ", ".join(d["sample_players"]))

    # 4) Colored 2D scatter of the clusters (PCA projection)
    # מציירים פיזור צבעוני של הקלאסטרים
    plot_clusters_2d(clustered, save_path=fig_dir / "10_kmeans_clusters.png")
    print(f"\nCluster scatter -> {fig_dir/'10_kmeans_clusters.png'}")


# נקודת כניסה
if __name__ == "__main__":
    main()
