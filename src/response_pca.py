"""
Collect LLM responses to beliefs from diverse personas and perform PCA
on the response matrix to measure belief covariance.
"""

import json
import os
import asyncio
import random
import numpy as np
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Diverse persona components for combinatorial generation
AGES = ["18-year-old", "25-year-old", "35-year-old", "45-year-old", "55-year-old", "65-year-old", "75-year-old"]
GENDERS = ["male", "female", "non-binary"]
EDUCATIONS = ["high school diploma", "bachelor's degree", "master's degree", "PhD", "no formal education"]
POLITICAL = [
    "far-left progressive", "liberal Democrat", "moderate centrist",
    "conservative Republican", "libertarian", "far-right nationalist",
    "apolitical", "green/environmentalist", "socialist", "populist"
]
RELIGIONS = [
    "devout Christian", "secular atheist", "agnostic", "Muslim",
    "Hindu", "Buddhist", "Jewish", "spiritual but not religious", "Sikh", "none"
]
COUNTRIES = [
    "United States", "United Kingdom", "India", "China", "Brazil",
    "Nigeria", "Japan", "Germany", "Mexico", "Saudi Arabia",
    "Sweden", "Russia", "South Korea", "Australia", "Egypt"
]
PERSONALITY = [
    "highly agreeable and empathetic",
    "skeptical and analytical",
    "traditional and conformist",
    "rebellious and independent-minded",
    "anxious and risk-averse",
    "optimistic and open to new experiences",
    "pragmatic and results-oriented",
    "idealistic and principled",
]


def generate_personas(n: int, seed: int = 42) -> list[str]:
    """Generate n diverse persona descriptions."""
    rng = random.Random(seed)
    personas = []
    seen = set()

    while len(personas) < n:
        p = {
            "age": rng.choice(AGES),
            "gender": rng.choice(GENDERS),
            "education": rng.choice(EDUCATIONS),
            "political": rng.choice(POLITICAL),
            "religion": rng.choice(RELIGIONS),
            "country": rng.choice(COUNTRIES),
            "personality": rng.choice(PERSONALITY),
        }
        key = tuple(p.values())
        if key not in seen:
            seen.add(key)
            desc = (f"a {p['age']} {p['gender']} from {p['country']} with "
                    f"{p['education']}, politically {p['political']}, "
                    f"religiously {p['religion']}, and generally {p['personality']}")
            personas.append(desc)

    return personas


async def rate_beliefs_batch(persona: str, beliefs: list[str], batch_id: int) -> list[int]:
    """Ask LLM to rate beliefs on 1-5 scale as a given persona."""
    belief_list = "\n".join(f"{i+1}. {b}" for i, b in enumerate(beliefs))

    prompt = f"""You are {persona}.

Rate your agreement with each of the following statements on a scale of 1-5:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

Respond with ONLY the numbers, one per line (just the rating, no explanation).
Stay in character as the persona described above.

Statements:
{belief_list}"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=len(beliefs) * 4,
        )
        text = response.choices[0].message.content.strip()
        ratings = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Extract number
            nums = [c for c in line if c.isdigit()]
            if nums:
                val = int(nums[-1])  # Take last digit in case of "1. 4" format
                if 1 <= val <= 5:
                    ratings.append(val)
                else:
                    ratings.append(3)  # Default to neutral
            # Stop if we have enough
            if len(ratings) >= len(beliefs):
                break

        # Pad if short
        while len(ratings) < len(beliefs):
            ratings.append(3)

        return ratings[:len(beliefs)]
    except Exception as e:
        print(f"  Error batch {batch_id}: {e}")
        return [3] * len(beliefs)


async def collect_responses(personas: list[str], beliefs: list[str],
                           beliefs_per_call: int = 50) -> np.ndarray:
    """Collect all persona responses to beliefs."""
    n_personas = len(personas)
    n_beliefs = len(beliefs)
    response_matrix = np.zeros((n_personas, n_beliefs), dtype=np.int8)

    sem = asyncio.Semaphore(40)
    total_calls = 0
    completed = 0

    async def process_persona(p_idx):
        nonlocal completed
        persona = personas[p_idx]
        all_ratings = []

        tasks = []
        for b_start in range(0, n_beliefs, beliefs_per_call):
            batch = beliefs[b_start:b_start + beliefs_per_call]
            batch_id = p_idx * 100 + b_start // beliefs_per_call

            async def do_call(p=persona, b=batch, bid=batch_id):
                async with sem:
                    return await rate_beliefs_batch(p, b, bid)

            tasks.append((b_start, do_call()))

        for b_start, task in tasks:
            ratings = await task
            end = min(b_start + beliefs_per_call, n_beliefs)
            response_matrix[p_idx, b_start:end] = ratings[:end - b_start]

        completed += 1
        if completed % 20 == 0:
            print(f"  Completed {completed}/{n_personas} personas")

    # Process all personas
    await asyncio.gather(*[process_persona(i) for i in range(n_personas)])

    return response_matrix


def analyze_response_pca(response_matrix: np.ndarray, beliefs: list[str]):
    """Run PCA on the response matrix and analyze."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_personas, n_beliefs = response_matrix.shape
    print(f"\nResponse matrix shape: {n_personas} personas x {n_beliefs} beliefs")
    print(f"Rating distribution: {np.bincount(response_matrix.flatten(), minlength=6)[1:]}")
    print(f"Mean rating: {response_matrix.mean():.2f}, Std: {response_matrix.std():.2f}")

    # Standardize columns (beliefs)
    scaler = StandardScaler()
    X = scaler.fit_transform(response_matrix.astype(float))

    # PCA on beliefs (columns) — compute covariance across personas
    n_components = min(n_personas - 1, n_beliefs)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nVariance explained (Response PCA):")
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n = np.searchsorted(cumvar, thresh) + 1
        if n <= len(cumvar):
            print(f"  {thresh*100:.0f}%: {n} components")
        else:
            print(f"  {thresh*100:.0f}%: >{len(cumvar)} components (max {cumvar[-1]*100:.1f}% with {len(cumvar)})")

    kaiser = np.sum(pca.explained_variance_ > 1.0)
    print(f"  Kaiser criterion: {kaiser} components")

    results = {
        "n_personas": n_personas,
        "n_beliefs": n_beliefs,
        "n_components": n_components,
        "rating_mean": float(response_matrix.mean()),
        "rating_std": float(response_matrix.std()),
        "variance_explained": {},
        "kaiser_criterion": int(kaiser),
        "top_20_eigenvalues": pca.explained_variance_[:20].tolist(),
        "top_20_variance_ratio": pca.explained_variance_ratio_[:20].tolist(),
        "cumulative_variance": cumvar.tolist(),
    }
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n = int(np.searchsorted(cumvar, thresh) + 1)
        results["variance_explained"][f"{int(thresh*100)}%"] = min(n, n_components)

    with open("results/response_pca_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---

    # 1. Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(50, n_components)
    ax = axes[0]
    ax.plot(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show], "bo-", markersize=4)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Explained Ratio")
    ax.set_title("Scree Plot (Response PCA)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, n_components + 1), cumvar, "r-", linewidth=2)
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        n = np.searchsorted(cumvar, thresh) + 1
        if n <= n_components:
            ax.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=n, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(f"{thresh*100:.0f}%: {n}",
                        xy=(n, thresh), fontsize=8, color="darkred")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance (Response PCA)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/response_pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Correlation heatmap of top beliefs
    corr = np.corrcoef(response_matrix.T)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title(f"Belief Correlation Matrix ({n_beliefs} beliefs)")
    ax.set_xlabel("Belief Index")
    ax.set_ylabel("Belief Index")
    plt.savefig("results/plots/belief_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Top component loadings
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for comp_idx, ax in enumerate(axes.flat):
        if comp_idx >= n_components:
            break
        loadings = pca.components_[comp_idx]
        top_pos = np.argsort(loadings)[-5:]
        top_neg = np.argsort(loadings)[:5]

        indices = np.concatenate([top_neg, top_pos])
        vals = loadings[indices]
        labels = [beliefs[i][:60] + "..." if len(beliefs[i]) > 60 else beliefs[i]
                  for i in indices]

        colors = ["red" if v < 0 else "steelblue" for v in vals]
        ax.barh(range(len(indices)), vals, color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"PC{comp_idx+1} ({pca.explained_variance_ratio_[comp_idx]*100:.1f}%)", fontsize=10)
        ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("results/plots/response_pca_loadings.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. 2D projection of personas
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=20, c="steelblue")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Personas in Belief Response PCA Space")
    ax.grid(True, alpha=0.3)
    plt.savefig("results/plots/persona_pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print top loading beliefs for first 3 components
    print("\n=== Top Loading Beliefs per Component ===")
    for comp_idx in range(min(5, n_components)):
        loadings = pca.components_[comp_idx]
        print(f"\nPC{comp_idx+1} ({pca.explained_variance_ratio_[comp_idx]*100:.1f}% variance):")
        top_pos = np.argsort(loadings)[-3:][::-1]
        top_neg = np.argsort(loadings)[:3]
        print("  Positive loadings:")
        for i in top_pos:
            print(f"    [{loadings[i]:+.3f}] {beliefs[i][:100]}")
        print("  Negative loadings:")
        for i in top_neg:
            print(f"    [{loadings[i]:+.3f}] {beliefs[i][:100]}")

    return results


async def main():
    random.seed(42)
    np.random.seed(42)

    # Load selected beliefs
    with open("results/selected_beliefs.json") as f:
        data = json.load(f)
    beliefs = data["beliefs"]
    print(f"Loaded {len(beliefs)} selected beliefs")

    # Generate personas
    personas = generate_personas(300)
    print(f"Generated {len(personas)} personas")
    print(f"Sample persona: {personas[0]}")

    # Check for cached responses
    if os.path.exists("results/response_matrix.npy"):
        print("Loading cached response matrix...")
        response_matrix = np.load("results/response_matrix.npy")
    else:
        print("Collecting LLM responses...")
        response_matrix = await collect_responses(personas, beliefs)
        np.save("results/response_matrix.npy", response_matrix)
        print("Saved response matrix")

    # Analyze
    analyze_response_pca(response_matrix, beliefs)


if __name__ == "__main__":
    asyncio.run(main())
