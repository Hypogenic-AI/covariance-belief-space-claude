# Covariance in Belief Space: PCA on 10,000 LLM-Generated Beliefs

## 1. Executive Summary

We generated 9,993 diverse belief statements and analyzed the covariance structure of "belief space" using three complementary methods: semantic embedding PCA (all 9,993 beliefs), LLM persona response PCA (300 beliefs × 300 personas), and conditional belief influence PCA (50 anchors × 100 test beliefs). **The key finding is that belief space is dramatically lower-dimensional than random.** Only **9 principal components** explain 50% of response variance, and **93 components** explain 90% — compared to 154 for random data. The dominant first component (28.6% of variance) maps cleanly to a progressive-conservative ideological axis. The conditional experiment reveals that only **25 dimensions** of belief influence explain 90% of how holding one belief affects agreement with others.

## 2. Goal

**Hypothesis**: Generating a diverse set of 10,000 beliefs and analyzing their covariance will reveal (a) which beliefs covary with each other, (b) that a small number of latent dimensions captures most of the variance, and (c) that these dimensions are interpretable.

**Importance**: Understanding belief dimensionality has implications for political science (how many independent opinion dimensions exist?), AI alignment (what latent value dimensions do LLMs encode?), survey design (how many questions do we really need?), and cognitive science (how are human belief systems structured?).

## 3. Data Construction

### Belief Generation
- **Source 1**: 1,226 existing belief questions from OpinionQA and curated datasets, yielding 2,452 perspective statements
- **Source 2**: 7,541 newly generated beliefs using GPT-4.1-mini across 50 topic categories × 20 angle modifiers
- **Final count**: 9,993 unique beliefs after deduplication
- **Topics covered**: politics, economics, religion, science, education, healthcare, environment, social justice, family, gender, immigration, criminal justice, military, media, art, food, housing, AI, mental health, drugs, animal rights, guns, free speech, death, tradition, nationalism, work, wealth, democracy, morality, parenting, aging, beauty, conspiracy, luck, punishment, privacy, competition, nature/nurture, freedom/security, and more

### Persona Generation
- 300 synthetic personas generated from combinatorial demographics:
  - 7 age groups × 3 genders × 5 education levels × 10 political orientations × 10 religions × 15 countries × 8 personality types
- Each persona rated 300 beliefs on a 1-5 Likert scale (Strongly Disagree to Strongly Agree)

### Data Quality
- Response matrix: 300 × 300, no missing values
- Rating distribution: heavily right-skewed (mean 3.97, std 1.23)
  - 1: 5,164 | 2: 10,304 | 3: 7,441 | 4: 26,693 | 5: 40,398
- This skew reflects that many beliefs were phrased as positive/aspirational statements

### Belief Selection for Response PCA
- 300 beliefs selected from 20 k-means clusters in embedding space
- 10 beliefs closest to each cluster center + 5 random per cluster
- Ensures maximal topical diversity in the subset

## 4. Experiment Description

### Methodology

We ran three complementary experiments:

#### Experiment 1: Semantic Embedding PCA
- Embedded all 9,993 beliefs using OpenAI `text-embedding-3-small` (1536 dimensions)
- Standardized and applied PCA to the full embedding matrix
- This measures **semantic** structure — which beliefs talk about similar topics

#### Experiment 2: Persona Response PCA
- 300 diverse personas each rated 300 beliefs via GPT-4.1-mini
- Built a 300 × 300 response matrix
- Standardized columns (beliefs) and applied PCA
- This measures **attitudinal covariance** — which beliefs tend to be held together

#### Experiment 3: Conditional Belief Influence
- Selected 50 anchor beliefs and 100 test beliefs
- For each anchor, asked: "Someone who strongly agrees with [anchor] — how would they rate [test beliefs]?" and the same for "strongly disagrees"
- Computed the difference matrix (agree - disagree ratings)
- PCA on the difference matrix measures **conditional influence** — how much knowing one belief predicts others

#### Why This Method?
Prior work (Ma & Powell 2025) showed LLMs can predict pairwise attitude correlations (r=0.77 with human data). We exploit this by using the LLM as a model of the human belief system, systematically sampling the full covariance structure rather than just pairwise correlations.

### Tools and Libraries
- Python 3.12, NumPy 1.26, scikit-learn 1.6, SciPy 1.17
- OpenAI API (gpt-4.1-mini for responses, text-embedding-3-small for embeddings)
- Matplotlib 3.10, Seaborn 0.13

### Reproducibility
- Random seed: 42 throughout
- All API calls used temperature=0.7 (personas) or 0.3 (conditional)
- Response matrix saved as `results/response_matrix.npy`
- All embeddings cached at `results/embeddings.npy`
- ~2,500 API calls for persona responses, ~200 for conditional experiment

## 5. Results

### Components Needed for Variance Thresholds

| Threshold | Semantic Embedding (9,993 beliefs) | Persona Response (300 beliefs) | Conditional Influence (50×100) | Random Baseline |
|-----------|-----------------------------------|-------------------------------|-------------------------------|-----------------|
| **50%** | 46 | **9** | **5** | 55 |
| **80%** | 200 | **56** | **16** | 117 |
| **90%** | 380 | **93** | **25** | 154 |
| **95%** | 501 | **127** | **32** | — |

**Kaiser criterion** (eigenvalue > 1): 251 (embedding), 60 (response)

### PC1: The Progressive-Conservative Axis (28.6% of response variance)

The dominant component captures a clear ideological dimension:

**Positive loadings (progressive/liberal):**
- "The criminal justice system ought to focus more on restorative justice" (+0.100)
- "Harm reduction programs like needle exchanges are essential" (+0.099)
- "National boundaries are socially constructed limitations" (+0.098)

**Negative loadings (conservative/traditional):**
- "Criminals must first be held accountable through punishment" (-0.097)
- "Wealth inequality reflects differences in individual effort and talent" (-0.097)
- "Police officers should be given broad authority to use force" (-0.097)

### PC2: Institutional Trust / Skepticism (5.8%)

**Positive**: Skepticism toward institutions, automation fears, neocolonialism concerns
**Negative**: Satisfaction with democracy, trust in private enterprise, positive view of markets

### PC3: Communitarianism vs. Individualism (4.6%)

**Positive**: Community duty, mutual obligation, tradition preservation
**Negative**: Tech-forward individualism, deprioritizing global reputation

### PC4: Spiritual/Metaphysical vs. Secular (3.6%)

**Positive**: Spiritual enlightenment, absolute spiritual truths
**Negative**: Secular/educational concerns, pragmatic perspectives

### PC5: Gender Equality Attitudes (2.2%)

**Positive/Negative**: Beliefs about women in leadership, gender barriers

### Conditional Influence Analysis

The conditional experiment reveals how holding one belief predicts others:

**Highest-impact anchor beliefs** (knowing agreement/disagreement most changes other ratings):
1. "Economic success is the only legitimate measure of merit" (impact: 2.07)
2. "Animal testing for cosmetics should be banned globally" (1.48)
3. "Countries should enforce absolute border closures" (1.47)

**Most sensitive beliefs** (most affected by which anchor is held):
1. "Attempts to ban high-capacity magazines infringe on rights" (sensitivity: 2.96)
2. "The state's primary role should be limited to protecting from force/fraud" (2.80)
3. "Global interconnectedness demands subordinating local self-interest" (2.74)

These high-sensitivity beliefs are the ones most "diagnostic" of someone's overall worldview.

### Key Visualizations

All plots saved in `results/plots/`:
- `comparison_cumvar.png`: Three PCA methods compared
- `response_vs_random.png`: Belief PCA vs random baseline
- `response_pca_scree.png`: Scree plot and cumulative variance
- `belief_correlation_heatmap.png`: Full 300×300 belief correlation matrix
- `response_pca_loadings.png`: Top loadings for first 6 components
- `persona_pca_2d.png`: Personas projected into PC1-PC2 space
- `conditional_belief_analysis.png`: Conditional influence matrix and PCA
- `embedding_pca_scree.png`: Embedding space scree plot
- `embedding_clusters.png`: 20 topic clusters in embedding space

## 5. Result Analysis

### Hypothesis Testing

**H1: Beliefs show strong covariance structure** — **CONFIRMED**
- Response PCA needs 93 components for 90% (vs 154 for random) — 40% fewer dimensions
- PC1 alone explains 28.6% (vs ~0.3% for random) — 95× more concentrated

**H2: Small number of components (<50) for 90%** — **PARTIALLY SUPPORTED**
- Conditional influence PCA: only 25 components for 90% — confirmed
- Response PCA: 93 components — more than 50 but still dramatically less than the 300 beliefs
- The difference reflects that response PCA captures fine-grained nuance beyond the major axes

**H3: Semantic and response PCA differ** — **CONFIRMED**
- Semantic embedding PCA is much higher-dimensional (380 components for 90%)
- Semantically dissimilar beliefs can covary strongly (e.g., gun rights and economic individualism)
- The gap between 380 (semantic) and 93 (response) shows that belief covariance is far simpler than topical diversity

**H4: Top components are interpretable** — **CONFIRMED**
- PC1: Progressive vs. conservative ideology (28.6%)
- PC2: Institutional trust vs. skepticism (5.8%)
- PC3: Communitarianism vs. individualism (4.6%)
- PC4: Spiritual vs. secular worldview (3.6%)
- PC5: Gender equality attitudes (2.2%)

These align with known dimensions from the World Values Survey (traditional-secular, survival-self-expression) and political science (left-right, libertarian-authoritarian).

### Comparison to Prior Work

- Ma & Powell (2025) found r=0.77 between LLM and human attitude correlations. Our PCA extends their pairwise analysis to reveal the full latent structure.
- WVS traditionally identifies 2 major value dimensions. Our analysis suggests ~5 major and ~25 significant dimensions for a broader belief set.
- Suh et al. (2025) found that "individual belief embeddings" cluster meaningfully. Our response PCA confirms this — personas naturally separate along ideological lines in PC space.

### Surprises

1. **The right-skew in ratings** (mean 3.97): LLMs tend to agree with most beliefs when playing personas, especially positively-framed ones. This may compress variance.
2. **PC1 dominance**: 28.6% for a single component is extremely strong, suggesting ideology is the "gravitational field" of belief space.
3. **Conditional vs. response dimensionality**: The conditional experiment (25 components for 90%) shows that the *causal* structure of beliefs is even simpler than the *correlational* structure (93 components).

### Limitations

1. **LLM as proxy for humans**: We used GPT-4.1-mini's model of beliefs, not actual human data. While Ma & Powell (2025) validated r=0.77 correspondence, the LLM may over-regularize belief structure.
2. **Persona diversity**: Despite combinatorial generation, LLM personas may not capture the full range of human belief combinations.
3. **300 beliefs subset**: Response PCA used 300 of 9,993 beliefs. Scaling to all 10K would require ~100× more API calls.
4. **Right-skewed ratings**: Many beliefs were positively framed, biasing toward agreement and potentially compressing variance.
5. **Single model**: Testing with Claude, Gemini, or open-source models would reveal whether dimensionality is model-dependent.

## 6. Conclusions

### Summary
Belief space, as modeled by LLMs, has dramatically lower dimensionality than its surface diversity suggests. While we generated ~10,000 topically diverse beliefs, only **9 principal components** capture 50% of how personas respond to them, and **93 capture 90%**. The conditional influence structure is even simpler: knowing someone's stance on a few key beliefs, just **25 dimensions** predict 90% of how they'll respond to others. The dominant axis (28.6% of variance) is a clear progressive-conservative dimension.

### Answer to the Research Question
**How many beliefs do you need to explain 90% of the variance?**
- In semantic space: ~380 components (high topical diversity)
- In attitudinal response space: **93 components** (moderate — beliefs cluster but retain nuance)
- In conditional influence space: **25 components** (low — the causal structure is sparse)

If you're trying to predict someone's beliefs from a questionnaire, ~25 well-chosen diagnostic questions would capture 90% of their belief profile.

### Implications
- **Survey design**: Most of the 10,000 beliefs are redundant. ~100 strategically chosen beliefs could reconstruct the full belief landscape.
- **AI alignment**: LLMs encode a structured, low-dimensional model of human values that can be probed and analyzed.
- **Political science**: The 5-component model (ideology, trust, individualism, spirituality, gender) provides a parsimonious framework for understanding opinion structure.

## 7. Next Steps

### Immediate Follow-ups
1. **Validate with human data**: Compare our PCA components to factor analyses from the World Values Survey and General Social Survey
2. **Cross-model comparison**: Run the same experiment with Claude and Gemini to test whether belief dimensionality is consistent
3. **Scale response PCA**: Use matrix completion or transfer learning to estimate the full 10K × 10K covariance from sparse sampling

### Alternative Approaches
- Nonlinear dimensionality reduction (t-SNE, UMAP, autoencoders) for belief structure
- Factor analysis with oblique rotation (allowing correlated factors)
- Hierarchical clustering to build a taxonomy of belief domains

### Open Questions
- Does belief dimensionality vary across cultures (conditioning on country)?
- Are there "bridge beliefs" that connect otherwise independent dimensions?
- How stable is the covariance structure over time (LLM version changes)?

## References

1. Ma, A., & Powell, D. (2025). "Can Large Language Models Predict Associations Among Human Attitudes?" arXiv:2503.21011
2. Santurkar, S., et al. (2023). "Whose Opinions Do Language Models Reflect?" ICML 2023. arXiv:2303.17548
3. Benkler, Y., et al. (2023). "Assessing LLMs for Moral Value Pluralism." NeurIPS 2023. arXiv:2312.10075
4. Suh, J., et al. (2025). "Language Model Fine-Tuning on Scaled Survey Data." arXiv:2502.16761
5. Rozado, D. (2025). "Measuring Political Preferences in AI Systems." arXiv:2503.10649
6. Karanjai, R., et al. (2025). "Synthesizing Public Opinions with LLMs." arXiv:2504.00241
