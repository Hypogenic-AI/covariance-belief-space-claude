"""Final comparison analysis and summary visualizations."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Load all results
    with open("results/embedding_pca_results.json") as f:
        emb_results = json.load(f)
    with open("results/response_pca_results.json") as f:
        resp_results = json.load(f)
    with open("results/conditional_analysis.json") as f:
        cond_results = json.load(f)

    # === Summary comparison figure ===
    fig, ax = plt.subplots(figsize=(10, 6))

    # Embedding PCA
    cumvar_emb = emb_results["cumulative_variance"]
    ax.plot(range(1, len(cumvar_emb) + 1), cumvar_emb,
            "b-", linewidth=2, label=f"Semantic Embeddings ({emb_results['n_beliefs']} beliefs)")

    # Response PCA
    cumvar_resp = resp_results["cumulative_variance"]
    ax.plot(range(1, len(cumvar_resp) + 1), cumvar_resp,
            "r-", linewidth=2, label=f"Persona Responses ({resp_results['n_beliefs']} beliefs)")

    # Conditional PCA - load from saved data
    agree = np.load("results/conditional_agree.npy")
    disagree = np.load("results/conditional_disagree.npy")
    diff = agree - disagree
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(diff)
    pca_cond = PCA(n_components=min(X.shape) - 1).fit(X)
    cumvar_cond = np.cumsum(pca_cond.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar_cond) + 1), cumvar_cond,
            "g-", linewidth=2, label="Conditional Differences (50 anchors × 100 test)")

    for thresh in [0.5, 0.8, 0.9]:
        ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.4)
        ax.annotate(f"{thresh*100:.0f}%", xy=(1, thresh + 0.01), fontsize=9, color="gray")

    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax.set_title("Dimensionality of Belief Space: Three Perspectives", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/comparison_cumvar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # === Summary table ===
    print("=" * 70)
    print("SUMMARY: Components needed for variance thresholds")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Embedding':<15} {'Response':<15} {'Conditional':<15}")
    print("-" * 57)

    for thresh_str, thresh_val in [("50%", 0.5), ("80%", 0.8), ("90%", 0.9), ("95%", 0.95)]:
        emb_n = emb_results["variance_explained"].get(thresh_str, "N/A")
        resp_n = resp_results["variance_explained"].get(thresh_str, "N/A")
        cond_n = int(np.searchsorted(cumvar_cond, thresh_val) + 1) if np.searchsorted(cumvar_cond, thresh_val) + 1 <= len(cumvar_cond) else "N/A"
        print(f"{thresh_str:<12} {str(emb_n):<15} {str(resp_n):<15} {str(cond_n):<15}")

    print()
    print(f"Kaiser criterion (eigenvalue > 1):")
    print(f"  Embedding: {emb_results['kaiser_criterion']}")
    print(f"  Response:  {resp_results['kaiser_criterion']}")

    print(f"\nPC1 variance explained:")
    print(f"  Embedding: {emb_results['top_10_variance_ratio'][0]*100:.1f}%")
    print(f"  Response:  {resp_results['top_20_variance_ratio'][0]*100:.1f}%")

    # === Random baseline ===
    np.random.seed(42)
    random_matrix = np.random.randint(1, 6, size=(300, 300))
    X_rand = StandardScaler().fit_transform(random_matrix.astype(float))
    pca_rand = PCA(n_components=min(X_rand.shape) - 1).fit(X_rand)
    cumvar_rand = np.cumsum(pca_rand.explained_variance_ratio_)

    print(f"\nBaseline comparison (random 300x300 matrix):")
    for thresh in [0.5, 0.8, 0.9]:
        n = int(np.searchsorted(cumvar_rand, thresh) + 1)
        print(f"  {thresh*100:.0f}%: {n} components")

    # Add random baseline to comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumvar_resp) + 1), cumvar_resp,
            "r-", linewidth=2.5, label=f"Belief Responses (LLM personas)")
    ax.plot(range(1, len(cumvar_cond) + 1), cumvar_cond,
            "g-", linewidth=2.5, label="Conditional Belief Influence")
    ax.plot(range(1, len(cumvar_rand) + 1), cumvar_rand,
            "k--", linewidth=1.5, alpha=0.5, label="Random Baseline (no structure)")

    for thresh in [0.5, 0.8, 0.9]:
        ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.3)

    # Annotate key points
    for t_val, label in [(0.9, "90%")]:
        n_resp = int(np.searchsorted(cumvar_resp, t_val) + 1)
        n_rand = int(np.searchsorted(cumvar_rand, t_val) + 1)
        ax.annotate(f"Response: {n_resp}", xy=(n_resp, t_val), fontsize=9,
                    color="red", xytext=(n_resp + 10, t_val - 0.05),
                    arrowprops=dict(arrowstyle="->", color="red"))
        ax.annotate(f"Random: {n_rand}", xy=(n_rand, t_val), fontsize=9,
                    color="black", xytext=(n_rand + 10, t_val + 0.03),
                    arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax.set_title("Belief Space Has Much Lower Dimensionality Than Random", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, min(200, len(cumvar_resp)))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/response_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save final summary
    summary = {
        "embedding_pca": {
            "n_beliefs": emb_results["n_beliefs"],
            "embedding_dim": emb_results["embedding_dim"],
            "components_for_90pct": emb_results["variance_explained"]["90%"],
            "kaiser": emb_results["kaiser_criterion"],
            "pc1_variance": emb_results["top_10_variance_ratio"][0],
        },
        "response_pca": {
            "n_personas": resp_results["n_personas"],
            "n_beliefs": resp_results["n_beliefs"],
            "components_for_90pct": resp_results["variance_explained"]["90%"],
            "kaiser": resp_results["kaiser_criterion"],
            "pc1_variance": resp_results["top_20_variance_ratio"][0],
        },
        "conditional_pca": {
            "n_anchors": cond_results["n_anchors"],
            "n_test_beliefs": cond_results["n_test_beliefs"],
            "components_for_90pct": cond_results["conditional_pca_variance"].get("90%"),
            "mean_abs_diff": cond_results["mean_abs_diff"],
        },
        "random_baseline": {
            "components_for_90pct": int(np.searchsorted(cumvar_rand, 0.9) + 1),
        },
        "key_finding": (
            f"Belief space is dramatically lower-dimensional than random. "
            f"Response PCA needs {resp_results['variance_explained']['90%']} components for 90% variance "
            f"(vs {int(np.searchsorted(cumvar_rand, 0.9) + 1)} for random). "
            f"The conditional experiment shows only {cond_results['conditional_pca_variance'].get('90%', 'N/A')} "
            f"dimensions of belief influence explain 90% of how beliefs affect each other. "
            f"PC1 ({resp_results['top_20_variance_ratio'][0]*100:.1f}%) maps to a progressive-conservative axis."
        ),
    }

    with open("results/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("KEY FINDING:")
    print(summary["key_finding"])
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
