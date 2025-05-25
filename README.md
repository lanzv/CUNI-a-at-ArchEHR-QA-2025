# CUNI-a at ArchEHR-QA 2025: Do we need Giant LLMs for Clinical QA?

This repository contains the code used in our submission to the BioNLP @ ACL 2025 ArchEHR-QA shared task described in our shared task paper **CUNI-a at ArchEHR-QA 2025: Do we need Giant LLMs for Clinical QA?**. The task focuses on answering patient questions based on electronic health record discharge summaries, without available training data.

In our submission, we explore whether accurate and relevant answers to patient questions can be generated without relying on large-scale LLMs. Our pipeline identifies essential sentences from the EHR using a combination of:
 - Few-shot inference with the Med42-8B model
 - Cosine similarity over clinical term embeddings
 - A cross-encoder relevance model (MedCPT)

Concise answers are then generated based on these selected sentences. Despite avoiding LLMs with tens of billions of parameters, our system performs competitively, highlighting the potential of resource-efficient clinical NLP solutions.

We compare two ensemble variants:
 - Ensemble-C, which combines methods based on ClinicalBERT-based cosine similarity
 - Ensemble-M, which combines methods based on mBERT-based cosine similarity instead


The table below compares our individual methods and their combinations, evaluated on the validation set for essential sentence retrieval.

| **Method**     | **F1** | *Precision* | *Recall* |
|----------------|--------|-------------|----------|
| All Sentences  | 48.8   | 32.2        | 100.0    |
| MCS-C          | 50.6   | 37.9        | 76.1     |
| MCS-M          | 48.9   | 32.6        | 97.8     |
| MedCPT FS      | 51.6   | 38.0        | 80.4     |
| MedCPT CT      | 51.8   | 44.3        | 62.3     |
| SR Med42       | 48.8   | 32.2        | 100.0    |
| CAR Med42      | 48.8   | 32.2        | 100.0    |
| Ensemble-C     | 56.8   | 53.2        | 60.9     |
| **Ensemble-M** |**58.6**| 52.3        | 66.7     |

## Usage

To generate answers for given dischare summary sentences and patient questions, run `generate_solution.py` script with the following arguments
 - `submission_file_path`: Output file path to save predicted answers to patient's questions
 - `dataset_file_path`: Path to the dataset `.xml` file (dev/test set)
 - `srmed_eps`: Threshold value for SR Med42
 - `cbmed_eps`: Threshold value for CAR Med42
 - `mces_eps`: Threshold value for MCS
 - `cptce_eps`: Threshold value for MedCPT-CT
 - `mces_model_path`: Path to the BERT/ClinicalBERT model
 - `medcptce_model_path`: Path to the MedCPT cross-encoder model
 - `med42_model_path`: Path to the Med42-8B model (few-shot inference)

### Ensemble-C Prediction

```bash
python generate_solution.py \
  --submission_file_path "./data/test/submission.json" \
  --dataset_file_path "./data/test/archehr-qa.xml" \
  --srmed_eps 0.0 \
  --cbmed_eps 0.4 \
  --mces_eps 0.9 \
  --cptce_eps 0.5 \
  --mces_model_path "../../models/Bio_ClinicalBERT" \
  --medcptce_model_path "../../models/MedCPT-Cross-Encoder" \
  --med42_model_path "../../models/Llama3-Med42-8B"
```

### Ensemble-M Prediction
```bash
python generate_solution.py \
  --submission_file_path "./data/test/submission.json" \
  --dataset_file_path "./data/test/archehr-qa.xml" \
  --srmed_eps 0.9 \
  --cbmed_eps 0.05 \
  --mces_eps 0.7 \
  --cptce_eps 0.5 \
  --mces_model_path "../../models/bert-base-multilingual-cased" \
  --medcptce_model_path "../../models/MedCPT-Cross-Encoder" \
  --med42_model_path "../../models/Llama3-Med42-8B"
```

## Experiments
`experiments.ipynb`: Explores different configurations and evaluates their impact on performance.

`visualization.ipynb`: Generates the charts and visualizations used in the paper.

## Citation

```bib
TODO

@inproceedings{lanz-pecina-2025-cuni-a,
    title = "CUNI-a at ArchEHR-QA 2025: Do we need Giant LLMs for Clinical QA?",
    author = "Lanz, Vojtech and Pecina, Pavel",
    booktitle = "Proceedings of the 24rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
}
```