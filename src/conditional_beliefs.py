"""
Test conditional belief covariance: put one belief in context,
measure how it affects agreement with other beliefs.
This directly tests "which beliefs covary with each other."
"""

import json
import os
import asyncio
import numpy as np
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def rate_beliefs_given_anchor(anchor_belief: str, test_beliefs: list[str],
                                      agree: bool) -> list[int]:
    """Rate test beliefs given agreement/disagreement with anchor."""
    stance = "strongly agree with" if agree else "strongly disagree with"
    belief_list = "\n".join(f"{i+1}. {b}" for i, b in enumerate(test_beliefs))

    prompt = f"""Someone who would {stance} the following statement:
"{anchor_belief}"

How would this person likely rate their agreement with each statement below?
Use a scale of 1-5 (1=Strongly Disagree, 5=Strongly Agree).
Respond with ONLY the numbers, one per line.

{belief_list}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=len(test_beliefs) * 4,
        )
        text = response.choices[0].message.content.strip()
        ratings = []
        for line in text.split("\n"):
            nums = [c for c in line.strip() if c.isdigit()]
            if nums:
                val = int(nums[-1])
                if 1 <= val <= 5:
                    ratings.append(val)
            if len(ratings) >= len(test_beliefs):
                break
        while len(ratings) < len(test_beliefs):
            ratings.append(3)
        return ratings[:len(test_beliefs)]
    except Exception as e:
        print(f"  Error: {e}")
        return [3] * len(test_beliefs)


async def run_conditional_experiment(beliefs: list[str], n_anchors: int = 50,
                                      n_test: int = 100):
    """
    For each anchor belief, get ratings when agreeing vs disagreeing.
    The difference reveals which beliefs are conditionally dependent.
    """
    np.random.seed(42)

    # Select anchor beliefs (spread across clusters)
    anchor_indices = np.random.choice(len(beliefs), n_anchors, replace=False)
    # Select test beliefs
    test_indices = np.random.choice(
        [i for i in range(len(beliefs)) if i not in anchor_indices],
        n_test, replace=False
    )

    anchor_beliefs = [beliefs[i] for i in anchor_indices]
    test_beliefs = [beliefs[i] for i in test_indices]

    sem = asyncio.Semaphore(30)

    # For each anchor: get ratings when agree and when disagree
    agree_matrix = np.zeros((n_anchors, n_test))
    disagree_matrix = np.zeros((n_anchors, n_test))

    completed = 0

    async def process_anchor(a_idx):
        nonlocal completed
        anchor = anchor_beliefs[a_idx]

        # Split test beliefs into batches
        batch_size = 50
        agree_ratings = []
        disagree_ratings = []

        for b_start in range(0, n_test, batch_size):
            batch = test_beliefs[b_start:b_start + batch_size]
            async with sem:
                a_r = await rate_beliefs_given_anchor(anchor, batch, agree=True)
            async with sem:
                d_r = await rate_beliefs_given_anchor(anchor, batch, agree=False)
            agree_ratings.extend(a_r)
            disagree_ratings.extend(d_r)

        agree_matrix[a_idx] = agree_ratings[:n_test]
        disagree_matrix[a_idx] = disagree_ratings[:n_test]

        completed += 1
        if completed % 10 == 0:
            print(f"  Completed {completed}/{n_anchors} anchors")

    await asyncio.gather(*[process_anchor(i) for i in range(n_anchors)])

    return anchor_indices, test_indices, agree_matrix, disagree_matrix


def analyze_conditional(anchor_indices, test_indices, agree_matrix, disagree_matrix,
                        beliefs):
    """Analyze conditional belief dependencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Difference matrix: how much does agreeing vs disagreeing change ratings
    diff_matrix = agree_matrix - disagree_matrix  # Shape: n_anchors x n_test

    print(f"\nConditional difference matrix stats:")
    print(f"  Shape: {diff_matrix.shape}")
    print(f"  Mean abs diff: {np.abs(diff_matrix).mean():.3f}")
    print(f"  Max diff: {diff_matrix.max():.1f}, Min diff: {diff_matrix.min():.1f}")
    print(f"  Std of diffs: {diff_matrix.std():.3f}")

    # Which test beliefs are most sensitive to anchor beliefs?
    sensitivity = np.std(diff_matrix, axis=0)  # Std across anchors for each test belief
    top_sensitive = np.argsort(sensitivity)[-10:][::-1]
    print(f"\nMost sensitive beliefs (high variance across anchors):")
    for idx in top_sensitive:
        print(f"  [{sensitivity[idx]:.2f}] {beliefs[test_indices[idx]][:100]}")

    # Which anchor beliefs create the biggest shifts?
    impact = np.mean(np.abs(diff_matrix), axis=1)
    top_impact = np.argsort(impact)[-10:][::-1]
    print(f"\nHighest impact anchor beliefs:")
    for idx in top_impact:
        print(f"  [{impact[idx]:.2f}] {beliefs[anchor_indices[idx]][:100]}")

    # PCA on the difference matrix to find latent conditional dimensions
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(diff_matrix)
    n_components = min(diff_matrix.shape) - 1
    pca = PCA(n_components=n_components)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nConditional Difference PCA:")
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        n = np.searchsorted(cumvar, thresh) + 1
        if n <= len(cumvar):
            print(f"  {thresh*100:.0f}%: {n} components")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(diff_matrix, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    plt.colorbar(im, ax=ax, label="Rating Diff (Agree - Disagree)")
    ax.set_xlabel("Test Belief Index")
    ax.set_ylabel("Anchor Belief Index")
    ax.set_title("Conditional Belief Influence Matrix")

    ax = axes[1]
    ax.plot(range(1, len(cumvar) + 1), cumvar, "r-", linewidth=2)
    for thresh in [0.5, 0.8, 0.9]:
        n = np.searchsorted(cumvar, thresh) + 1
        if n <= len(cumvar):
            ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=n, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(f"{thresh*100:.0f}%: {n}", xy=(n, thresh), fontsize=9, color="darkred")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("PCA on Conditional Differences")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/conditional_belief_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Covariance heatmap of most correlated beliefs
    corr = np.corrcoef(diff_matrix.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Covariance Between Test Beliefs (Conditional)")
    plt.savefig("results/plots/conditional_covariance.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "n_anchors": len(anchor_indices),
        "n_test_beliefs": len(test_indices),
        "mean_abs_diff": float(np.abs(diff_matrix).mean()),
        "conditional_pca_variance": {
            f"{int(t*100)}%": int(np.searchsorted(cumvar, t) + 1)
            for t in [0.5, 0.8, 0.9, 0.95]
            if np.searchsorted(cumvar, t) + 1 <= len(cumvar)
        },
        "top_impact_anchors": [
            {"belief": beliefs[anchor_indices[i]][:150], "impact": float(impact[i])}
            for i in top_impact[:5]
        ],
        "top_sensitive_beliefs": [
            {"belief": beliefs[test_indices[i]][:150], "sensitivity": float(sensitivity[i])}
            for i in top_sensitive[:5]
        ],
    }

    with open("results/conditional_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


async def main():
    with open("results/selected_beliefs.json") as f:
        data = json.load(f)
    beliefs = data["beliefs"]
    print(f"Loaded {len(beliefs)} beliefs")

    print("Running conditional belief experiment...")
    anchor_idx, test_idx, agree_mat, disagree_mat = await run_conditional_experiment(
        beliefs, n_anchors=50, n_test=100
    )

    np.save("results/conditional_agree.npy", agree_mat)
    np.save("results/conditional_disagree.npy", disagree_mat)

    analyze_conditional(anchor_idx, test_idx, agree_mat, disagree_mat, beliefs)


if __name__ == "__main__":
    asyncio.run(main())
