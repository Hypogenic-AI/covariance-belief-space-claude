# Cloned Repositories

## Repo 1: OpinionQA
- **URL**: https://github.com/tatsu-lab/opinions_qa
- **Purpose**: Benchmark for evaluating LLM opinion alignment with 60 U.S. demographic groups
- **Location**: code/opinions_qa/
- **Key files**:
  - `process_results.ipynb` — Compute human and LM opinion distributions
  - `representativeness.ipynb` — Analyze human-LM alignment
  - `steerability.ipynb` — Test demographic prompting effectiveness
  - `helpers.py` — Utility functions for data processing
- **Notes**: Data must be downloaded separately from [CodaLab](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69). The HuggingFace version (timchen0618/OpinionQA) provides a subset with 1176 questions.

## Repo 2: WorldValuesBench
- **URL**: https://github.com/Demon702/WorldValuesBench
- **Purpose**: Benchmark dataset for studying multi-cultural human value awareness of language models, derived from World Values Survey Wave 7
- **Location**: code/WorldValuesBench/
- **Key files**:
  - `dataset_construction/data_preparation.py` — Process raw WVS data into benchmark format
  - `dataset_construction/question_metadata.json` — Metadata for 240 value questions
  - `dataset_construction/codebook.json` — Mapping between numerical answers and natural language
  - `evaluation/evaluate.py` — Earth Mover's Distance evaluation script
  - `WorldValuesBench/` — Generated benchmark data (requires raw WVS data first)
- **Notes**:
  - Raw WVS data must be downloaded from [WVS website](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) (free registration required)
  - Contains 93,278 participants, 240 value questions, 42 demographic variables
  - Includes train/valid/test splits (70/15/15)
  - The question_metadata.json and codebook.json are immediately useful without raw data
