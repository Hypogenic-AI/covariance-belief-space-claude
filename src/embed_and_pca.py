"""Embed all beliefs and perform PCA on the embedding space."""

import json
import numpy as np
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def get_embeddings_batch(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    response = await client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


async def embed_all(beliefs: list[str], batch_size=500) -> np.ndarray:
    """Embed all beliefs in batches."""
    sem = asyncio.Semaphore(20)
    all_embeddings = [None] * len(beliefs)

    async def process_batch(start_idx):
        async with sem:
            batch = beliefs[start_idx:start_idx + batch_size]
            embs = await get_embeddings_batch(batch)
            for i, emb in enumerate(embs):
                all_embeddings[start_idx + i] = emb
            if start_idx % 2000 == 0:
                print(f"  Embedded {start_idx + len(batch)}/{len(beliefs)}")

    tasks = [process_batch(i) for i in range(0, len(beliefs), batch_size)]
    await asyncio.gather(*tasks)

    return np.array(all_embeddings)


def run_pca_analysis(embeddings: np.ndarray, beliefs: list[str]):
    """Run PCA and generate analysis."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # Full PCA
    n_components = min(500, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nVariance explained (embedding PCA):")
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh*100:.0f}%: {n} components")

    # Kaiser criterion
    kaiser_count = np.sum(pca.explained_variance_ > 1.0)
    print(f"  Kaiser criterion (eigenvalue > 1): {kaiser_count} components")

    # Save results
    results = {
        "n_beliefs": len(beliefs),
        "embedding_dim": embeddings.shape[1],
        "n_components_analyzed": n_components,
        "variance_explained": {
            "50%": int(np.searchsorted(cumvar, 0.5) + 1),
            "80%": int(np.searchsorted(cumvar, 0.8) + 1),
            "90%": int(np.searchsorted(cumvar, 0.9) + 1),
            "95%": int(np.searchsorted(cumvar, 0.95) + 1),
            "99%": int(np.searchsorted(cumvar, 0.99) + 1),
        },
        "kaiser_criterion": int(kaiser_count),
        "top_10_eigenvalues": pca.explained_variance_[:10].tolist(),
        "top_10_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
        "cumulative_variance": cumvar.tolist(),
    }

    with open("results/embedding_pca_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---

    # 1. Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(1, min(51, n_components + 1)),
            pca.explained_variance_ratio_[:50], "bo-", markersize=3)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Explained Ratio")
    ax.set_title("Scree Plot (Embedding PCA, top 50)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, n_components + 1), cumvar, "r-", linewidth=2)
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        n = np.searchsorted(cumvar, thresh) + 1
        ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=n, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(f"{thresh*100:.0f}%: {n}",
                    xy=(n, thresh), fontsize=8, color="darkred")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance (Embedding PCA)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/embedding_pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.1, s=1, c="steelblue")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("10K Beliefs in Embedding PCA Space (PC1 vs PC2)")
    ax.grid(True, alpha=0.3)
    plt.savefig("results/plots/embedding_pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Identify clusters using k-means on PCA space
    from sklearn.cluster import KMeans

    n_clusters = 20
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca[:, :50])

    fig, ax = plt.subplots(figsize=(12, 8))
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.2, s=2, label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("20 Belief Clusters in Embedding Space")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    plt.savefig("results/plots/embedding_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print cluster examples
    print("\n--- Cluster Examples ---")
    for c in range(min(5, n_clusters)):
        indices = np.where(labels == c)[0]
        samples = np.random.choice(indices, min(3, len(indices)), replace=False)
        print(f"\nCluster {c} ({np.sum(labels==c)} beliefs):")
        for idx in samples:
            print(f"  - {beliefs[idx][:100]}")

    # Save cluster assignments and select representative beliefs
    np.save("results/embedding_pca_components.npy", X_pca)
    np.save("results/cluster_labels.npy", labels)

    # Select diverse beliefs for response-based PCA
    # Pick beliefs closest to each cluster center + random from each
    selected_indices = []
    for c in range(n_clusters):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]
        cluster_points = X_pca[cluster_mask, :50]
        center = km.cluster_centers_[c]

        # Closest to center
        dists = np.linalg.norm(cluster_points - center, axis=1)
        sorted_idx = np.argsort(dists)

        # Take 10 closest + 5 random
        n_close = min(10, len(sorted_idx))
        n_random = min(5, max(0, len(sorted_idx) - n_close))
        close = cluster_indices[sorted_idx[:n_close]]
        selected_indices.extend(close)
        if n_random > 0:
            remaining = cluster_indices[sorted_idx[n_close:]]
            rand_pick = np.random.choice(remaining, n_random, replace=False)
            selected_indices.extend(rand_pick)

    selected_indices = sorted(set(selected_indices))
    print(f"\nSelected {len(selected_indices)} diverse beliefs for response PCA")

    selected_beliefs = [beliefs[i] for i in selected_indices]
    with open("results/selected_beliefs.json", "w") as f:
        json.dump({"indices": [int(i) for i in selected_indices], "beliefs": selected_beliefs}, f, indent=2)

    return results


async def main():
    # Load beliefs
    with open("results/beliefs_10k.json") as f:
        beliefs = json.load(f)

    print(f"Loaded {len(beliefs)} beliefs")

    # Check if embeddings already computed
    if os.path.exists("results/embeddings.npy"):
        print("Loading cached embeddings...")
        embeddings = np.load("results/embeddings.npy")
    else:
        print("Computing embeddings...")
        embeddings = await embed_all(beliefs)
        np.save("results/embeddings.npy", embeddings)
        print(f"Saved embeddings: {embeddings.shape}")

    # Run PCA analysis
    run_pca_analysis(embeddings, beliefs)


if __name__ == "__main__":
    asyncio.run(main())
