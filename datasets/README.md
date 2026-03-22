# Downloaded Datasets

This directory contains datasets for the Covariance in Belief Space research project. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: OpinionQA (HuggingFace)

### Overview
- **Source**: [timchen0618/OpinionQA on HuggingFace](https://huggingface.co/datasets/timchen0618/OpinionQA)
- **Original Source**: Pew American Trends Panel surveys via [Santurkar et al. 2023](https://arxiv.org/abs/2303.17548)
- **Size**: 1,176 questions (882 test + 294 validation)
- **Format**: HuggingFace Dataset (Arrow format)
- **Task**: Opinion/attitude survey questions with multiple perspectives
- **License**: Research use

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("timchen0618/OpinionQA")
dataset.save_to_disk("datasets/opinionqa")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/opinionqa")
# Each item has: question, perspectives, id
for item in dataset['test']:
    print(item['question'], item['perspectives'])
```

### Sample Data
See `opinionqa/samples.json` for 20 example questions.

### Notes
- Questions cover diverse topics: safety, politics, technology, social issues, economics, etc.
- Each question has 2+ perspectives (agree/disagree framings)
- Originally derived from 1498 Pew survey questions (full dataset available via CodaLab)

---

## Dataset 2: Curated Belief Statements

### Overview
- **Source**: Compiled from OpinionQA, World Values Survey items, and curated diverse beliefs
- **Size**: 1,226 belief statements
- **Format**: JSON
- **File**: `belief_statements.json`
- **Breakdown**:
  - OpinionQA: 1,176 questions
  - World Values Survey items: 20 statements
  - Curated diverse beliefs: 30 statements

### Loading the Dataset

```python
import json
with open("datasets/belief_statements.json") as f:
    beliefs = json.load(f)
# Each item has: question, perspectives, id, source
```

### Notes
- This is the primary dataset for the experiment
- WVS items cover traditional-secular value dimensions with known factor loadings
- Curated items fill gaps in coverage (technology, environment, philosophy)
- For the experiment, a subset of ~100-200 diverse items should be selected for the covariance analysis

---

## Dataset 3: World Values Survey (Reference Only)

### Overview
- **Source**: [World Values Survey Wave 7](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp)
- **Size**: 93,278 participants, 240 value questions, 100+ nations
- **Format**: CSV (available after registration)
- **License**: Free for non-profit research use, registration required

### Download Instructions

1. Navigate to the [WVS Wave 7 website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp)
2. Go to `Statistical Data Files` section
3. Click on `WVS Cross-National Wave 7 csv v6 0.zip`
4. Fill out the form (name, institution, purpose)
5. The raw CSV data will be automatically downloaded

### Notes
- Registration required — cannot be automatically downloaded
- The WorldValuesBench repo (code/WorldValuesBench) provides processing scripts
- Known factor structure: traditional-secular axis (factor loadings 0.51-0.70)
- Key value dimensions: Inglehart-Welzel cultural map

---

## Dataset 4: SubPOP (Gated — Reference Only)

### Overview
- **Source**: [jjssuh/subpop on HuggingFace](https://huggingface.co/datasets/jjssuh/subpop)
- **Size**: 3,362 questions, 70K subpopulation-response pairs
- **Format**: HuggingFace Dataset (gated access)
- **License**: Research use, requires access approval

### Download Instructions

```python
# Requires HuggingFace authentication and access approval
from datasets import load_dataset
dataset = load_dataset("jjssuh/subpop")
```

### Notes
- Gated dataset — requires requesting access on HuggingFace
- Contains individual belief embeddings that are directly relevant to our research
- From the paper "Language Model Fine-Tuning on Scaled Survey Data" (arXiv:2502.16761)
