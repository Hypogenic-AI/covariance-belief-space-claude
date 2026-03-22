# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Covariance in Belief Space" research project, including papers, datasets, and code repositories. The research hypothesis is that generating 10,000 diverse beliefs and analyzing their covariance via PCA will reveal which beliefs covary and how many principal components explain 90% of the variance.

## Papers
Total papers downloaded: 15

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Can LLMs Predict Associations Among Human Attitudes? | Ma, Powell | 2025 | papers/2503.21011_*.pdf | r=0.77 LLM-human attitude correlation |
| Whose Opinions Do Language Models Reflect? | Santurkar et al. | 2023 | Via code repo | OPINIONQA benchmark, 1498 questions |
| Measuring Political Preferences in AI Systems | Rozado | 2025 | papers/2503.10649_*.pdf | Multi-method political bias assessment |
| Assessing LLMs for Moral Value Pluralism | Benkler et al. | 2023 | papers/2312.10075_*.pdf | WVS factor analysis, NeurIPS 2023 |
| LM Fine-Tuning on Survey Data | Suh et al. | 2025 | papers/2502.16761_*.pdf | Belief embeddings, SubPOP dataset |
| Belief Space Planning: Covariance Steering | Zheng et al. | 2021 | papers/2105.11092_*.pdf | Robotics belief space (different domain) |
| Synthesizing Public Opinions with LLMs | Karanjai et al. | 2025 | papers/2504.00241_*.pdf | Role creation with HEXACO model |
| LLM Moral Hypocrites | Various | 2024 | papers/2405.11100_*.pdf | Moral foundations in LLMs |
| FairBelief | Various | 2024 | papers/2402.17389_*.pdf | Harmful beliefs assessment |
| Specializing LLMs for Survey Distributions | Various | 2025 | papers/2502.07068_*.pdf | Population-level opinion simulation |
| AI-Augmented Surveys | Various | 2023 | papers/2305.09620_*.pdf | LLM+survey opinion prediction |
| Validating Opinion Dynamics | Banisch, Shamon | 2022 | papers/2212.10143_*.pdf | Survey experiment validation |
| LLMs Grasp Morality | Various | 2023 | papers/2311.02294_*.pdf | Moral concept representation |
| FAST-PCA | Gang, Bajwa | 2021 | papers/2108.12373_*.pdf | Distributed PCA algorithm |
| TL-PCA | Hendy, Dar | 2024 | papers/2410.10805_*.pdf | Transfer learning for PCA |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2 (+ 2 reference datasets requiring registration)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| OpinionQA | HuggingFace | 1,176 questions | Opinion survey | datasets/opinionqa/ | Downloaded, ready to use |
| Curated Belief Statements | Compiled | 1,226 statements | Belief analysis | datasets/belief_statements.json | Ready to use |
| World Values Survey | WVS website | 93K participants | Cross-cultural values | Reference only | Requires registration |
| SubPOP | HuggingFace (gated) | 3,362 questions | Opinion prediction | Reference only | Requires access approval |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| OpinionQA | github.com/tatsu-lab/opinions_qa | OPINIONQA benchmark code | code/opinions_qa/ | Notebooks for analysis |
| WorldValuesBench | github.com/Demon702/WorldValuesBench | WVS benchmark for LLMs | code/WorldValuesBench/ | Data processing pipeline |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** was attempted first but returned empty results
2. **arXiv API** searched with 16 query combinations across belief, covariance, PCA, opinion, survey, LLM, and moral foundations topics — yielded 137 unique papers
3. **Web search** used for finding datasets (HuggingFace, Pew, WVS, GSS) and code repositories
4. **Citation chaining** from the most relevant paper (Ma & Powell 2025) led to foundational works (Santurkar 2023, Hwang 2023, Argyle 2023)

### Selection Criteria
- **Papers**: Prioritized work that (a) directly studies belief/attitude covariance, (b) uses LLMs to generate/predict opinions, (c) applies PCA/factor analysis to belief data, or (d) provides datasets of belief statements
- **Datasets**: Focused on collections of diverse opinion/belief questions with human response data for validation
- **Code**: Prioritized repos with data processing pipelines and evaluation frameworks

### Challenges Encountered
- SubPOP dataset is gated on HuggingFace — requires access approval
- World Values Survey requires registration (free for non-profit) — cannot be auto-downloaded
- The full OPINIONQA dataset requires download from CodaLab (the HuggingFace version has 1176/1498 questions)
- No existing codebase directly implements PCA on LLM-generated belief responses

### Gaps and Workarounds
- **No direct PCA-on-beliefs codebase exists** — experiment runner will need to implement this from scratch using sklearn/numpy
- **No pre-generated 10,000 belief dataset** — experiment runner will generate beliefs using LLM API calls
- **Ground truth for validation** — WVS factor structure (traditional-secular, survival-self-expression) provides known dimensions to validate against

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**:
   - Use `belief_statements.json` (1,226 statements) as the source of belief questions
   - Select a diverse subset of 100-200 statements covering all major topic areas
   - For each of 10,000 "personas," generate responses (1-5 Likert scale) to all selected beliefs

2. **Belief generation method**:
   - Generate diverse personas using demographic combinations (age, gender, education, political leaning, religion, nationality) + personality traits (Big Five or HEXACO)
   - Prompt an LLM to respond to all belief statements as each persona
   - Collect responses into a 10,000 × N_beliefs matrix

3. **Analysis pipeline**:
   - Compute correlation/covariance matrix of the belief response matrix
   - Apply PCA and report: (a) scree plot, (b) cumulative variance explained curve, (c) number of components for 90% variance
   - Analyze top component loadings for interpretability
   - Compare with known factor structures (political left-right, WVS traditional-secular)

4. **Baseline methods**:
   - Random responses (null model)
   - PCA on human survey data (if available, e.g., from OPINIONQA)
   - Factor analysis (oblique rotation) as alternative to PCA

5. **Evaluation metrics**:
   - Number of components for 90% variance explained
   - Kaiser criterion (eigenvalue > 1)
   - Scree plot elbow
   - Component interpretability (alignment with known dimensions)

6. **Code to adapt/reuse**:
   - `code/opinions_qa/` — for processing survey questions and computing distributions
   - `code/WorldValuesBench/` — for question metadata and codebook
   - Standard sklearn PCA pipeline for dimensionality reduction
