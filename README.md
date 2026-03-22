# Covariance in Belief Space

PCA analysis on 10,000 LLM-generated beliefs to determine the dimensionality of "belief space."

## Key Findings

- **9,993 diverse beliefs** generated across 50+ topics using GPT-4.1-mini
- **9 components** explain 50% of variance in how personas respond to beliefs
- **93 components** explain 90% of variance (vs 154 for random — 40% fewer)
- **25 conditional dimensions** explain 90% of how holding one belief predicts others
- **PC1 (28.6%)** is a clear progressive-conservative ideological axis
- Top 5 interpretable dimensions: ideology, institutional trust, communitarianism, spirituality, gender attitudes

## Quick Start

```bash
# Setup
source .venv/bin/activate
export OPENAI_API_KEY=your_key

# Generate beliefs
python src/generate_beliefs.py

# Embed and PCA on semantic space
python src/embed_and_pca.py

# Collect persona responses and PCA on response space
python src/response_pca.py

# Conditional belief influence experiment
python src/conditional_beliefs.py

# Final analysis and comparison plots
python src/final_analysis.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan
├── src/
│   ├── generate_beliefs.py    # Generate 10K beliefs via LLM
│   ├── embed_and_pca.py       # Embedding PCA (semantic space)
│   ├── response_pca.py        # Persona response PCA (attitudinal space)
│   ├── conditional_beliefs.py # Conditional belief influence experiment
│   └── final_analysis.py      # Comparison analysis and summary
├── results/
│   ├── beliefs_10k.json       # All 9,993 generated beliefs
│   ├── embeddings.npy         # 9993 × 1536 embedding matrix
│   ├── response_matrix.npy    # 300 × 300 persona-belief ratings
│   ├── selected_beliefs.json  # 300 diverse beliefs for response PCA
│   ├── *_results.json         # PCA results for each experiment
│   └── plots/                 # All visualizations
├── datasets/                  # Pre-gathered datasets (OpinionQA, curated beliefs)
├── papers/                    # Reference papers
└── literature_review.md       # Literature review
```

## Full Report

See [REPORT.md](REPORT.md) for methodology, results, and analysis.
