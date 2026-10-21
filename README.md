# Responsible Medical LLMs for Mental Healthcare

**LCS2, IIT Delhi — Coding Round**

This repository implements a controlled evaluation of two instruction-tuned language models on the MedMisBench-MentalHealth benchmark. The experiments assess four dimensions of responsible medical LLM behavior: hallucinations and uncertainty, interpretability and explainability, bias and fairness, and privacy and security.

## 1. Models and Dataset

### Models

The evaluation uses the two models specified in the assignment:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Both models are evaluated using local Hugging Face inference on a Kaggle GPU. The models are run sequentially rather than loaded simultaneously.

### Dataset

- **Dataset:** [Johndfm/MedMisBench-MentalHealth](https://huggingface.co/datasets/Johndfm/MedMisBench-MentalHealth)
- **Dataset revision:** `9757095`
- **Source file/split:** `medmisbench_mentalhealth.jsonl`, `train`
- **Local dataset file:** `medmisbench_mentalhealth.json`

The experiments use the fixed subsets defined in the assignment. Rows are indexed in their original stored order, starting from zero, without shuffling.

| Subset | Size | Purpose |
|---|---:|---|
| Core | 200 items | Clean baseline evaluation |
| Stress | 60 items | Hallucination, fairness, privacy/security, and cross-dimension experiments |
| Interpretability | 40 items | Internal-model and input-evidence analysis |

## 2. Repository Structure

```text
.
├── analysis/
│   ├── 4_baseline.ipynb
│   ├── 5_1_hallicunation.ipynb
│   ├── 5_2_interpretability.ipynb
│   ├── 5_3_bias.ipynb
│   ├── 5_4_prompt_canary.ipynb
│   ├── 6_cross_analysis.ipynb
│   └── 7_intervention.ipynb
├── meta_csv/
│   └── Meta-model CSV outputs
├── qwen_csv/
│   └── Qwen-model CSV outputs
├── iitd-coding-round.ipynb
├── medmisbench_mentalhealth.json
├── requirements.txt
├── time_per_subtask.csv
└── README.md
```

### Main files and directories

- **`iitd-coding-round.ipynb`** — Main Kaggle inference notebook. Run all cells to execute the model inference workflow and save the resulting CSV files in the corresponding model directories.

- **`analysis/`** — Contains seven Jupyter notebooks for baseline evaluation, the four evaluation dimensions, cross-dimension analysis, and the mitigation experiment.

- **`meta_csv/`** — Stores CSV outputs for `meta-llama/Llama-3.2-3B-Instruct`.

- **`qwen_csv/`** — Stores CSV outputs for `Qwen/Qwen2.5-3B-Instruct`.

- **`medmisbench_mentalhealth.json`** — Local copy of the benchmark dataset used by the project.

- **`requirements.txt`** — Records Python dependencies and package versions for future reproduction and environment consistency.

- **`time_per_subtask.csv`** — Records the time taken by a model for each subtask.

## 3. Environment and Dependencies

The experiments were developed and executed in a Kaggle notebook with GPU acceleration. All required packages were already available in the Kaggle environment, so no additional package installation was necessary for the original runs.

The `requirements.txt` file is retained to document the dependency environment and support future reproduction. For a fresh environment, install the listed dependencies using:

```bash
pip install -r requirements.txt
```

For the closest reproduction of the original results, use compatible package versions and a GPU environment supported by the selected models.

## 4. Inference Configuration

The following generation settings are used for model inference:

| Setting | Value |
|---|---|
| Model precision | Float16 (`float16`) |
| `do_sample` | `False` |
| `max_new_tokens` | `512` |
| Random seed for preprocessing/sampling | `42` |
| Model execution | Sequential, one model at a time |
| Compute environment | Kaggle GPU |

The exact resolved model revisions and any additional model-specific configuration should be taken from the main inference notebook and its saved outputs.

### Fixed date for Llama

The Llama model's output can include a date, which introduces an avoidable source of variation in the generated text. To keep this part of the prompt consistent, the inference code uses the fixed value:

```python
date_string = "20 Oct 2026"
```

This date is intentionally fixed rather than generated dynamically at runtime. It ensures that the date component of the prompt remains unchanged across runs.

**Reproducibility note:** `do_sample=False` disables sampling, making generation greedy under the configured decoding setup. It does not guarantee byte-for-byte identical output across every hardware and software environment. The fixed date controls the date-related prompt variation but does not, by itself, guarantee complete determinism.

## 5. Reproducing the Experiments

### Step 1: Open the main inference notebook

Open `iitd-coding-round.ipynb` in Kaggle and enable GPU acceleration.

Ensure that access to both Hugging Face model repositories is available. The Meta Llama model may require accepting its access conditions and configuring authentication.

### Step 2: Run the inference notebook

Run all cells in `iitd-coding-round.ipynb`.

The notebook executes the inference workflow for both models and saves the generated CSV files in `meta_csv/` and `qwen_csv/`.

The notebook should be run using the specified dataset, model revisions, fixed subsets, prompts, and generation settings. Do not change these settings when attempting to reproduce the reported experiments.

### Step 3: Run the analysis notebooks

After the required inference outputs have been generated, run the notebooks in `analysis/` in the following order:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `4_baseline.ipynb` | Baseline evaluation |
| 2 | `5_1_hallicunation.ipynb` | Hallucinations and uncertainty under misleading context |
| 3 | `5_2_interpretability.ipynb` | Interpretability and explainability |
| 4 | `5_3_bias.ipynb` | Bias and fairness under metadata counterfactuals |
| 5 | `5_4_prompt_canary.ipynb` | Targeted prompt injection and synthetic-canary leakage |
| 6 | `6_cross_analysis.ipynb` | Relationships across evaluation dimensions |
| 7 | `7_intervention.ipynb` | Before-and-after evaluation of the mitigation |

Each notebook analyzes the corresponding experiment using the generated CSV outputs. If a notebook has dependencies on results produced by an earlier notebook, execute it after those dependencies have been generated.

### Step 4: Verify the outputs

After running the experiments, check that:

- Both model-specific directories contain the expected CSV outputs.
- Each condition contains the expected number of records for its fixed subset.
- Raw outputs, parsed answers, confidence values, and other required fields are preserved where applicable.
- Invalid or failed generations are retained rather than silently omitted.
- The results reported in the paper agree with the machine-readable outputs.

### Step 5: Inspect runtime records

Use `time_per_subtask.csv` to examine the recorded model runtime for each subtask. Runtime may vary with GPU availability, software versions, and execution conditions.
