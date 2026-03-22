# Research Plan: Covariance in Belief Space

## Motivation & Novelty Assessment

### Why This Research Matters
Human beliefs do not exist in isolation — they form structured, correlated systems. Understanding the dimensionality of "belief space" reveals how many independent axes of opinion actually exist, which has implications for political science, survey design, AI alignment, and understanding LLM representations of human values.

### Gap in Existing Work
Ma & Powell (2025) showed LLMs can predict pairwise attitude correlations (r=0.77), and Suh et al. (2025) introduced "belief embeddings" that cluster meaningfully. However, **no study has explicitly computed PCA on a large-scale LLM-generated belief covariance matrix** to determine the intrinsic dimensionality of belief space. Prior work used 64-1498 items; we scale to 10,000.

### Our Novel Contribution
1. Generate 10,000 diverse belief statements — the largest belief set analyzed for covariance
2. Perform PCA in two complementary spaces: semantic (embedding) space and response (agreement) space
3. Determine how many principal components explain 90% of variance in each space
4. Identify interpretable belief clusters and their covariance structure

### Experiment Justification
- **Experiment 1 (Belief Generation)**: Need 10K diverse beliefs to map the full space
- **Experiment 2 (Embedding PCA)**: Fast, scalable analysis of semantic structure across all 10K beliefs
- **Experiment 3 (Response PCA)**: Tests actual belief covariance (which beliefs are held together) using LLM as a model of human belief systems
- **Experiment 4 (Comparison)**: Validates whether semantic similarity predicts belief covariance

## Research Question
How many latent dimensions are needed to explain 90% of the variance in a space of 10,000 diverse beliefs? What is the covariance structure of belief space?

## Hypothesis Decomposition
1. Beliefs will show strong covariance structure (not independent)
2. A small number of components (<50) will explain 90% of variance in response space
3. Semantic embedding PCA and response-based PCA will identify partially overlapping but distinct dimensions
4. Top components will be interpretable (mapping to known dimensions like political ideology, religiosity, etc.)

## Proposed Methodology

### Approach
1. **Generate 10,000 beliefs** using GPT-4.1 with diverse topic prompts
2. **Embed all beliefs** using text-embedding-3-small (1536-dim)
3. **PCA on embeddings** — semantic structure analysis
4. **Select 300 diverse beliefs** via k-means clustering on embeddings
5. **Generate 200 diverse personas** spanning demographics and ideologies
6. **Rate beliefs per persona** using gpt-4.1-mini (batch 50 beliefs per call)
7. **PCA on response matrix** (200 personas × 300 beliefs) — belief covariance analysis
8. **Compare and interpret** both PCA results

### Baselines
- Random responses (null model — no covariance structure)
- Single-ideology responses (maximum covariance — all beliefs correlated)
- Known factor structures (WVS traditional-secular, political left-right)

### Evaluation Metrics
- Cumulative variance explained at 50%, 80%, 90%, 95%
- Number of components at each threshold
- Scree plot eigenvalue decay
- Component interpretability (top-loading beliefs per component)

### Statistical Analysis Plan
- Kaiser criterion (eigenvalue > 1) for component count
- Parallel analysis for significance
- Bootstrap confidence intervals on variance explained
- Correlation between embedding PCA and response PCA structures

## Timeline
- Phase 1-2: Setup & belief generation (~30 min)
- Phase 3: Embedding PCA (~15 min)
- Phase 4: LLM response collection (~60 min)
- Phase 5: Response PCA & analysis (~30 min)
- Phase 6: Documentation (~20 min)

## Success Criteria
- Successfully generate 10,000 diverse beliefs
- Determine number of components for 90% variance
- Identify interpretable principal components
- Compare semantic vs. response-based dimensionality
