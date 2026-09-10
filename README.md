# Toxic Language Detection on Social Media

**An End-to-End NLP Pipeline Using Classical Machine Learning and Transformer-Based Models**

MSc Computing Research Project — Data Science & Artificial Intelligence, Sheffield Hallam University

**Author:** Sai Harsha Papisetti (35054998)

**Supervisor:** Dr Osagie Efosa

---

## Overview

Automated toxic language detection is essential for content moderation at social-media scale, yet existing research disproportionately benchmarks short-form text and rarely reports classification accuracy alongside computational resource cost or algorithmic fairness. This project addresses that gap with a single, reproducible pipeline that benchmarks four classical machine learning classifiers against a fine-tuned DistilBERT transformer for detecting toxicity in **long-sequence** (150+ word) social media comments — jointly reporting accuracy, resource cost, and fairness under identical conditions.

**Research Question:** How does an end-to-end NLP pipeline comparing classical ML and transformer-based models perform in detecting toxic language in long-sequence social media text?

## Objectives

1. Conduct a critical literature review of toxic language detection, focusing on long-text classification and resource-efficient NLP architectures.
2. Collect, preprocess, and filter the Jigsaw dataset (160K+ records) to isolate long-sequence text.
3. Implement and evaluate four classical ML classifiers.
4. Fine-tune a resource-efficient transformer (DistilBERT) on the same data.
5. Compare all models on F1, ROC-AUC, precision, recall, and latency.
6. Conduct a computational algorithmic fairness audit — no human participants (UREC1 ethics scope).

## Pipeline

The project follows a fixed, five-phase pipeline so every model is compared under identical conditions:

```
Data Pre-Processing → Classical Baselines → Transformer Fine-Tuning → Model Evaluation → Fairness Audit
```

| Phase | Description |
|---|---|
| 1. Data Pre-Processing | Binary label collapse, long-sequence filtering, 80/10/10 stratified split |
| 2. Classical Baselines | TF-IDF + Logistic Regression, Linear SVM, Naive Bayes, Random Forest |
| 3. Transformer Fine-Tuning | DistilBERT fine-tuned via Hugging Face Transformers |
| 4. Model Evaluation | F1, ROC-AUC, precision, recall, training time, inference latency |
| 5. Fairness Audit | Computational subgroup analysis using dialectal proxy markers |

## Dataset

- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) (Kaggle)
- **Licence:** CC0-1.0 (public domain)
- **Size:** 159,571 records, six overlapping toxicity labels collapsed into one binary target
- **Long-sequence filter:** 150+ words → 16,353 records (10.17% → 6.29% toxic prevalence)
- **Split:** 80/10/10, stratified on the binary label (13,081 train / 1,636 validation / 1,636 test)

No primary data was collected. This project is classified under **UREC1 (No Human Participants)** — see [Ethics](#ethics--data-protection).

## Results

All models trained and evaluated on the identical held-out test set (n = 1,636):

| Model | F1 | ROC-AUC | Precision | Recall | Train Time (s) | Inference Time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.597 | 0.948 | 0.510 | 0.718 | 0.46 | 0.0008 |
| Linear SVM | 0.642 | 0.945 | 0.624 | 0.660 | 0.40 | 0.0008 |
| Naive Bayes | 0.306 | 0.888 | 0.905 | 0.184 | 0.01 | 0.0011 |
| Random Forest | 0.561 | 0.945 | 0.815 | 0.427 | 29.70 | 0.2093 |
| **DistilBERT** | **0.768** | **0.969** | 0.800 | 0.738 | 296.39 | 3.0687 |

DistilBERT outperforms the best classical baseline (Linear SVM) by **12.6 percentage points of F1**, at roughly **490× the training cost** and a disproportionately larger inference-latency cost — the key engineering trade-off this project identifies.

### Fairness Audit

DistilBERT's F1-score, computed separately on test comments containing informal dialectal proxy markers:

| Subgroup | N | F1-score | Precision | Recall |
|---|---|---|---|---|
| Overall | 1,636 | 0.7677 | 0.8000 | 0.7379 |
| Marker present | 105 | 0.5714 | 0.7500 | 0.4615 |
| Marker absent | 1,531 | 0.7910 | 0.8046 | 0.7778 |

The gap is driven by **recall, not precision** — the model under-flags this subgroup rather than over-flagging it, a reversal of the over-flagging direction typically reported in the fairness literature (Sap et al., 2019). See the full report for discussion.

## Repository Structure

```
├── notebooks/
│   └── toxic_language_pipeline.ipynb    # Full pipeline: all 5 phases + result figures
├── models/                              # Saved model artefacts (.joblib / HF checkpoint)
├── results/
│   ├── model_comparison_results.csv
│   └── fairness_audit_results.csv
├── figures/                             # Generated charts (model comparison, latency, fairness)
├── docs/
│   └── report.docx                      # Full dissertation report
└── README.md
```

## Getting Started

### Requirements

- Python 3.12
- Google Colab (T4 GPU recommended) or local environment with CUDA
- Kaggle account (for dataset download via API)

### Setup

1. Clone this repository:
```bash
   git clone https://github.com/<your-username>/toxic-language-detection.git
```
2. Open `notebooks/toxic_language_pipeline.ipynb` in Google Colab.
3. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.
4. Upload your Kaggle API token (`kaggle.json`) when prompted, or manually place `train.csv` in `/content/data/`.
5. Run all cells top to bottom.

### Key Dependencies

```
transformers
datasets
accelerate
scikit-learn
pandas
matplotlib
torch
```

## Experimental Configuration

- **Hardware:** Single NVIDIA T4 GPU (16GB VRAM), Google Colaboratory
- **Software:** Python 3.12, PyTorch 2.x, Hugging Face Transformers/Datasets, scikit-learn 1.x
- **Reproducibility:** Fixed random seed (`seed = 42`) across all data splitting, shuffling, and model initialisation
- **DistilBERT fine-tuning:** 3 epochs, batch size 16, max sequence length 256 tokens, AdamW optimiser, mixed-precision (fp16)

## Ethics & Data Protection

This project uses **only secondary, publicly available, pre-anonymised data** (Jigsaw dataset, CC0-1.0 licence). No primary data was collected and no human participants were recruited at any stage, including the fairness audit, which is conducted entirely computationally on existing dataset text.

Classified under **UREC1 — Research Ethics Review for Student Research with No Human Participants**, Sheffield Hallam University. See `docs/report.docx` (Appendix B) for the full signed ethics form.

## Limitations

- Single platform (Wikipedia talk pages) and single language (English) — generalisability to other platforms/languages is untested.
- The fairness audit uses a simple five-term lexical proxy for dialectal markers, not a validated sociolinguistic classifier, on a small subgroup (n = 105).
- Only DistilBERT was fine-tuned within the project timeline; BERT-base was not run, so the hypothesised near-parity between the two remains untested.
- Model validation relied on supervisor review rather than end-user/participant feedback, consistent with the UREC1 (no-participants) ethical scope.

## AI Declaration

AI tools (Claude, Anthropic) assisted with code drafting and debugging, dissertation chapter structuring, and documentation formatting throughout this project. All research questions, design decisions, analysis, interpretation of results, and final judgements are the author's own.

## References

Key sources — full APA 7th edition reference list in `docs/report.docx`:

- Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *ACL 2020*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL 2019*.

## License

This project uses the Jigsaw Toxic Comment Classification dataset (CC0-1.0). Code in this repository is provided for academic purposes as part of an MSc dissertation at Sheffield Hallam University.

## Contact

Sai Harsha Papisetti — C5054998@hallam.shu.ac.uk
