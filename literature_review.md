# Literature Review: Covariance in Belief Space

## Research Area Overview

This research investigates the structure of human beliefs by generating a diverse set of beliefs using LLMs and analyzing their covariance structure via PCA. The key question is: **which beliefs covary with each other, and how many latent dimensions are needed to explain 90% of the variance in belief space?**

This topic sits at the intersection of several active research areas:
1. **LLM opinion simulation** — using language models to generate synthetic survey responses
2. **Belief system structure** — understanding how attitudes cluster and correlate
3. **Dimensionality reduction of attitudes** — applying PCA/factor analysis to opinion data
4. **Cultural value measurement** — standardized surveys like WVS and Pew that measure values cross-culturally

## Key Papers

### Paper 1: Can Large Language Models Predict Associations Among Human Attitudes?
- **Authors**: Ana Ma, Derek Powell
- **Year**: 2025
- **Source**: arXiv:2503.21011
- **Key Contribution**: Demonstrates that GPT-4o can recreate pairwise correlations among 64 diverse human attitudes (r=0.77 with human data), even for semantically dissimilar beliefs.
- **Methodology**: Surveyed 376 U.S. adults on 64 attitude statements from OPINIONQA. Prompted GPT-4o with one attitude+response to predict another. Compared LLM-estimated correlations with human correlations.
- **Datasets Used**: Custom survey of 64 items drawn from OPINIONQA/Pew data, 376 participants.
- **Results**: GPT-4o recreated attitude correlations strongly (r=0.77). Even filtering for only semantically dissimilar pairs (cosine similarity < 0.20), correlation remained strong (r=0.724). GPT-4o achieved ~43-45% accuracy predicting individual attitudes (vs 29.4% chance).
- **Code Available**: No
- **Relevance to Our Research**: **CRITICAL** — This paper directly validates that LLMs capture the latent covariance structure of human belief systems. Their finding that LLMs can predict attitude associations beyond surface similarity suggests LLMs encode a compressed representation of belief space. Our research extends this by explicitly computing the covariance matrix and applying PCA.

### Paper 2: Whose Opinions Do Language Models Reflect? (OPINIONQA)
- **Authors**: Santurkar, Durmus, Ladhak, Lee, Liang, Hashimoto
- **Year**: 2023
- **Source**: arXiv:2303.17548 (ICML 2023)
- **Key Contribution**: Introduced OPINIONQA benchmark with 1498 survey questions from 15 Pew American Trends Panel surveys covering 60 U.S. demographic groups.
- **Methodology**: Probed LLMs with survey questions, compared response distributions to human demographic groups along representativeness, steerability, and consistency axes.
- **Datasets Used**: Pew American Trends Panel data (80,098 respondents, 1506 questions, 15 surveys).
- **Results**: LLMs showed left-leaning political bias; responses most aligned with liberal, college-educated demographics. Steerability via demographic prompting was limited.
- **Code Available**: Yes — github.com/tatsu-lab/opinions_qa
- **Relevance to Our Research**: **HIGH** — Provides the foundation dataset (OPINIONQA) and methodology for probing LLM opinions. The 1498 questions span diverse topics making them ideal for belief covariance analysis.

### Paper 3: Assessing LLMs for Moral Value Pluralism
- **Authors**: Benkler, Mosaphir, Friedman, Smart, Schmer-Galunder
- **Year**: 2023
- **Source**: arXiv:2312.10075 (NeurIPS 2023)
- **Key Contribution**: Used World Values Survey (WVS) value dimensions to assess implicit moral values in LLM-generated text using a Recognizing Value Resonance (RVR) model.
- **Methodology**: Prompted LLMs to take the point-of-view of various demographics (nationality, age, sex). Used RVR model to compute resonance of LLM outputs with WVS values along traditional-secular axis. Factor loadings from WVS used (0.51-0.70 range).
- **Datasets Used**: World Values Survey Wave 7 (100+ nations, 40 years), traditional-secular value items.
- **Results**: LLMs showed WEIRD bias — misaligned with non-Western nations. Age misalignment across nations was also significant.
- **Code Available**: No
- **Relevance to Our Research**: **HIGH** — Demonstrates that beliefs cluster along known factor-analytic dimensions (traditional vs. secular). The WVS factor analysis provides ground truth for validating whether our PCA recovers similar dimensions.

### Paper 4: Measuring Political Preferences in AI Systems
- **Authors**: David Rozado
- **Year**: 2025
- **Source**: arXiv:2503.10649
- **Key Contribution**: Multi-method assessment of political bias in LLMs using (1) linguistic comparison with Congress members, (2) policy recommendation analysis, (3) sentiment toward political figures, (4) standardized political tests.
- **Methodology**: Four complementary methodologies integrated into aggregated bias index. Tested base, conversational, and ideologically-aligned LLMs.
- **Results**: Consistent left-leaning bias across most AI systems. Bias is not inherent — fine-tuning with politically skewed data can shift models across the spectrum.
- **Code Available**: No
- **Relevance to Our Research**: **MODERATE** — Confirms that LLMs have structured political biases, which will manifest as dominant dimensions in our PCA analysis.

### Paper 5: Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions
- **Authors**: Suh et al.
- **Year**: 2025
- **Source**: arXiv:2502.16761
- **Key Contribution**: Fine-tuned LLMs on SubPOP dataset (3,362 questions, 70K subpopulation-response pairs) to predict response distributions. Introduced individual belief embeddings.
- **Methodology**: Incorporated individual belief embeddings to account for heterogeneous responses. Random latent features for each individual optimized during fine-tuning — individuals close in embedding space have similar belief sets.
- **Results**: Reduced LLM-human gap by up to 46% compared to baselines. Strong generalization to unseen surveys.
- **Code Available**: Dataset on HuggingFace (jjssuh/subpop, gated)
- **Relevance to Our Research**: **HIGH** — The concept of "individual belief embeddings" is directly related to our belief space. Their finding that belief embeddings cluster meaningfully suggests a low-dimensional belief space structure.

### Paper 6: Synthesizing Public Opinions with LLMs
- **Authors**: Karanjai, Shor, Austin, Kennedy, Lu, Xu, Shi
- **Year**: 2025
- **Source**: arXiv:2504.00241
- **Key Contribution**: Role creation based on knowledge injection using HEXACO personality model and demographic information. Improved LLM-generated opinion alignment with human survey responses.
- **Methodology**: RAG-enhanced prompting with personality profiles and demographics. Tested on Cooperative Election Study (CES) questions.
- **Results**: Significantly improved answer adherence over standard few-shot prompting.
- **Code Available**: No
- **Relevance to Our Research**: **MODERATE** — Provides techniques for generating more diverse and realistic belief profiles, which we can use to generate our 10,000 belief samples.

### Paper 7: Belief Space Planning: A Covariance Steering Approach
- **Authors**: Zheng, Ridderhof, Tsiotras, Agha-mohammadi
- **Year**: 2021
- **Source**: arXiv:2105.11092
- **Key Contribution**: Covariance steering in robotics belief space for motion planning under uncertainty.
- **Methodology**: Uses covariance steering theory to control the state distribution of a robot. Nodes represent Gaussian distributions; edges steer between distributions.
- **Results**: CS-BRM achieves finite-time belief node reachability, overcoming limitations of stationary node sampling.
- **Code Available**: No
- **Relevance to Our Research**: **LOW** — Different domain (robotics). Uses "belief space" in the POMDP/estimation sense (probability distributions over robot states), not human beliefs/opinions. However, the mathematical framework of covariance steering provides interesting parallels.

### Paper 8: AI-Augmented Surveys
- **Authors**: Multiple authors
- **Year**: 2023 (updated 2024)
- **Source**: arXiv:2305.09620
- **Key Contribution**: Leveraging LLMs and surveys for opinion prediction, addressing curse of dimensionality in survey research.
- **Relevance**: **MODERATE** — Provides context for using LLMs to augment opinion data collection.

### Paper 9: Validating Argument-Based Opinion Dynamics with Survey Experiments
- **Authors**: Banisch, Shamon
- **Year**: 2022
- **Source**: arXiv:2212.10143
- **Key Contribution**: Empirical validation of computational opinion formation models using survey experiments.
- **Relevance**: **MODERATE** — Provides methodology for comparing model-generated opinion dynamics with empirical data.

## Common Methodologies
- **Survey prompting of LLMs**: Used in Papers 1, 2, 3, 4, 5, 6 — prompting LLMs with demographic info or prior attitudes to predict survey responses
- **Correlation/covariance analysis**: Papers 1, 2 — computing pairwise correlations among attitude responses
- **Factor analysis / PCA**: Paper 3 (WVS), Paper 5 (belief embeddings) — reducing high-dimensional belief data to latent factors
- **Cosine similarity**: Paper 1 — measuring semantic similarity between belief statements
- **Wasserstein distance**: WorldValuesBench — comparing model vs human response distributions

## Standard Baselines
- **Random/chance baseline**: ~29.4% for 5-choice prediction (Paper 1)
- **Cosine similarity matching**: Predicting based on most semantically similar attitude
- **Demographic-conditioned prompting**: Using demographics to steer LLM responses (Paper 2)
- **Oracle random forest**: Upper bound using trained models on human data (Paper 1: ~47-50%)

## Evaluation Metrics
- **Pearson correlation**: Between LLM-estimated and human attitude correlations (Paper 1: r=0.77)
- **Prediction accuracy**: Exact match between LLM output and human response
- **Earth Mover's Distance (Wasserstein-1)**: Between model and human response distributions (WorldValuesBench)
- **Variance explained**: PCA cumulative variance (our primary metric — targeting 90%)
- **Cultural distance**: Between LLM values and WVS survey demographics (Paper 3)

## Datasets in the Literature
- **OPINIONQA**: 1498 questions from 15 Pew American Trends Panel surveys, 80K respondents (Papers 1, 2)
- **SubPOP**: 3,362 questions, 70K subpopulation-response pairs from public opinion surveys (Paper 5)
- **World Values Survey Wave 7**: 93,278 participants, 240 value questions, 100+ nations (Paper 3, WorldValuesBench)
- **Cooperative Election Study (CES)**: U.S. political attitudes survey (Paper 6)
- **General Social Survey (GSS)**: Decades of U.S. social attitudes data
- **Moral Foundations Questionnaire**: Measures 5-6 moral foundations

## Gaps and Opportunities
1. **No study directly computes PCA on LLM-generated belief covariance matrix** — Prior work analyzes correlations (Paper 1) or uses belief embeddings (Paper 5), but none explicitly extract the principal components of belief space and report variance explained.
2. **Scale of belief generation**: Prior work uses 64 items (Paper 1) or ~1500 items (Paper 2) but doesn't generate 10,000 diverse beliefs to comprehensively map the space.
3. **Cross-model comparison**: No study compares belief space dimensionality across different LLMs.
4. **Explicit dimensionality question**: No paper directly answers "how many components explain 90% of belief variance?"

## Recommendations for Our Experiment

### Recommended Datasets
1. **OPINIONQA** (1176 questions available on HuggingFace) — primary source for belief statements
2. **World Values Survey items** — add cross-cultural value dimensions
3. **Curated diverse beliefs** — supplementary statements covering underrepresented topics
4. **Self-generated beliefs** — use LLM to generate additional diverse belief statements to reach 10,000

### Recommended Baselines
1. **Random responses** — beliefs generated uniformly at random
2. **Single-ideology responses** — beliefs from a single consistent political/moral perspective
3. **Human survey data** — if available, compare LLM belief covariance with human belief covariance (as in Paper 1)

### Recommended Metrics
1. **Cumulative variance explained by PCA** — primary metric (target: 90%)
2. **Number of principal components for 90% variance** — key finding
3. **Scree plot** — eigenvalue decay pattern
4. **Correlation matrix visualization** — heatmap of belief covariances
5. **Component loadings** — which beliefs load on which components (interpretability)

### Methodological Considerations
- **Belief generation strategy**: Use diverse persona prompts (demographics, personality, ideology) to generate responses, as validated by Papers 1, 5, 6
- **Response scale**: Use 5-point Likert (Strongly Disagree to Strongly Agree) as in Paper 1
- **Sample size**: 10,000 belief profiles is sufficient for stable PCA on ~100-1000 variables
- **Multiple LLMs**: Consider testing with multiple models (GPT-4, Claude, etc.) to see if dimensionality varies
- **Validation**: Compare discovered factors with known dimensions (left-right political axis, traditional-secular from WVS, moral foundations)
