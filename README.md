# Knowledge Graph Embedding-Based Performance Prediction and Dataset Similarity

This repository contains the codebase accompanying the paper [**Integrating Meta-Features with Knowledge Graph Embeddings for Meta-Learning**]([https://example.org/paper.pdf](https://link.springer.com/chapter/10.1007/978-3-032-25156-5_18) by Klironomos A., Dasoulas I., Periti F., Gad-Elrab M., Paulheim H., Dimou A., Kharlamov E., accepted at ESWC 2026.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Command Line Arguments](#command-line-arguments)
- [Reproducing Paper Results](#reproducing-paper-results)
  - [Step 0: Generate ExeKGs from OpenML Runs](#step-0-generate-exekgs-from-openml-runs)
  - [Step 1: Prepare Data](#step-1-prepare-data)
  - [Step 2: Train RDF2Vec KGE Models](#step-2-train-rdf2vec-kge-models)
  - [Downstream Task 1: Pipeline Performance Estimation (PPE)](#downstream-task-1-pipeline-performance-estimation-ppe)
    - [Step 2a: Train Link Prediction Models (Optional)](#step-2a-train-link-prediction-models-optional)
    - [Step 3: Pipeline Performance Prediction](#step-3-pipeline-performance-prediction)
    - [Step 4: Plot Pipeline Performance Prediction Results](#step-4-plot-pipeline-performance-prediction-results)
  - [Downstream Task 2: Dataset Performance Similarity Estimation (DPSE)](#downstream-task-2-dataset-performance-similarity-estimation-dpse)
    - [Step 5: Calculate KGE-Based Similarities](#step-5-calculate-kge-based-similarities)
    - [Step 6: Calculate Graph Edit Distances (GEDs)](#step-6-calculate-graph-edit-distances-geds)
    - [Step 7: Plot Dataset Similarity Results](#step-7-plot-dataset-similarity-results)
  - [Step 8: MLSeaKG Ablation Study](#step-8-mlseakg-ablation-study)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Repository Structure

The repository consists of two main subprojects:

1. **`openml_exekgs_generation/`**: Crawls OpenML to obtain experiment data and generates executable knowledge graphs (ExeKGs) from runs and tasks. This subproject handles the data collection and initial KG generation phase.

2. **`kge_experiments/`**: The main subproject that trains Knowledge Graph Embeddings and uses them for two downstream meta-learning tasks.

   **Core functionality includes:**
   - Preparing and processing knowledge graph data
   - Training RDF2Vec-based Knowledge Graph Embedding (KGE) models
   - Training Link Prediction (LP) models on the knowledge graph (optional, for PPE task only)
   - Analyzing results and generating publication-ready figures and tables
   
   **Downstream Task 1: Pipeline Performance Estimation (PPE)**
   - Predicting pipeline performance on datasets using RDF2Vec or LP embeddings
   - Evaluating performance prediction accuracy on unseen datasets and pipelines
   
   **Downstream Task 2: Dataset Performance Similarity Estimation (DPSE)**
   - Calculating KGE-based dataset similarities using RDF2Vec embeddings
   - Computing Graph Edit Distances (GEDs) between datasets
   - Evaluating dataset similarity measures for retrieving similar datasets

## Prerequisites

- Python 3.10
- Poetry (for dependency management)

Install dependencies using Poetry:  
```bash
poetry install
```

## Command Line Arguments

This section describes the command-line arguments for both subprojects.

### Arguments for `openml_exekgs_generation/`

These arguments are used when generating ExeKGs from OpenML (Step 0).

- `--mp-runs`: Enable multiprocessing for runs (default: False)
- `--mp-tasks`: Enable multiprocessing for tasks (default: False)
- `--task-id <List[int]>`: Specify particular task IDs to process (optional)
- `--use-processed-tasks-and-runs`: Use previously processed tasks and runs (default: False)
- `--offset-step <int>`: Step size for processing tasks in batches (default: 1000)
- `--n-tasks <int>`: Number of tasks to process (optional, processes all if not specified)
- `--n-runs-per-flow <int>`: Number of runs to process per flow (default: 10)

---

### Arguments for `kge_experiments/`

These arguments are used for training KGE models and performing meta-learning tasks (Steps 1+).

#### Global Arguments

The following arguments can be used with any `kge_experiments` command:

- `--verbose`: Enable verbose output for detailed logging and debugging information
- `--rdf2vec-d <int>`: Set the distance (depth) of random walks for RDF2Vec embeddings (default: 10)
- `--rdf2vec-w <int>`: Set the number of random walks per entity for RDF2Vec embeddings (default: 10)  
- `--rdf2vec-ws <string>`: Set the walk strategy for RDF2Vec embeddings (default: "random")
- `--use-mlseakg`: Include MLSeaKG (Machine Learning Semantic Knowledge Graph) in addition to ExeKGs
- `--use-mkga`: Use MKGA (Multi-Modal Knowledge Graph Augmentation) preprocessing method

#### Command-Specific Arguments

##### `prepare-data`

- `--filter-by-dataset-ids`: Filter the data to include only specific dataset IDs (predefined in the code)
- `--excl-flows-per-task`: Exclude invalid flows per task based on predefined criteria
- `--excl-performance-values`: Exclude performance values from the knowledge graph (for PPE downstream task)
- `--remove-old-files`: Remove existing processed files before generating new ones

##### `train-rdf2vec`

- `--chunk-size <int>`: Set the chunk size for processing data in batches (default: 100)
- `--cpu-count <int>`: Set the number of CPU cores to use for parallel processing (default: 4)

##### `train-link-prediction`

- `--model-type <string>`: Type of link prediction model to train - options: DistMultReaLitEGated, TransEReaLitEGated, DistMultReaLitE, TransEReaLitE, ComplExReaLitEGated, ComplExReaLitE (default: "DistMultReaLitEGated")
- `--timeout-hours <float>`: Timeout in hours for training (optional)
- `--inverse-triples` / `--no-inverse-triples`: Create inverse triples in the knowledge graph (default: enabled, should be True for LP models)

*Note: Link Prediction models are used exclusively for the PPE downstream task. These models handle literals natively, so MKGA preprocessing (--use-mkga) should be disabled when training LP models.*

##### `calculate-kge-similarities`

- `--replace-similarities`: Replace existing similarity calculations if they already exist
- `--replace-info`: Replace existing dataset information if it already exists
- `--add-data-entity-sim`: Calculate similarities based on data entity embeddings only
- `--add-data-entity-and-pipeline-sim`: Calculate similarities using both data entity and pipeline embeddings
- `--add-pipeline-sim`: Calculate similarities based on pipeline embeddings only
- `--add-dataset-info`: Add dataset information to the similarity results

*Note: At least one of the similarity calculation options (--add-data-entity-sim, --add-data-entity-and-pipeline-sim, --add-pipeline-sim, or --add-dataset-info) must be specified.*

##### `calculate-and-add-geds-to-similarities`

- `--replace-geds`: Replace existing Graph Edit Distance (GED) calculations if they already exist
- `--add-dataset-info`: Add dataset information to the GED results
- `--replace-info`: Replace existing dataset information in GED results if it already exists
- `--num-processes <int>`: Set the number of parallel processes for GED calculation (default: 10)
- `--chunk-size <int>`: Set the chunk size for processing GED calculations in batches (default: 5)

##### `predict-pipeline-performance`

- `--split-mode <string>`: Set the data splitting strategy - either "dataset" or "pipeline" (default: "dataset")
- `--target <string>`: Specify the target variable to predict (default: "f1_score")
- `--emb-aggr-type <string>`: Set embedding aggregation type - either "concat" or "mean" (default: "concat")
- `--model <string>`: Choose the machine learning model - options include RF, SVC, LR, RFReg, SVR, LRReg (default: "RF")
- `--dataset-emb-source <string>`: Specify the dataset embedding source - "rdf2vec", "pykeen-lp", or metafeature types (metafeatures_all, metafeatures_statistical, metafeatures_landmarkers, metafeatures_information_theory) (default: "rdf2vec")
- `--pipeline-emb-source <string>`: Specify the pipeline embedding source - "rdf2vec" or "pykeen-lp" (default: "rdf2vec")
- `--min-train-samples-per-run-id <int>`: Set minimum training samples per run ID (default: 50)
- `--pykeen-lp-model-name <string>`: PyKEEN LP model name (required when using pykeen-lp as embedding source) - options: DistMultReaLitEGated, TransEReaLitEGated, DistMultReaLitE, TransEReaLitE, ComplExReaLitEGated, ComplExReaLitE
- `--pykeen-lp-inverse-triples` / `--no-pykeen-lp-inverse-triples`: Whether the PyKEEN model was trained with inverse triples (default: enabled, must match training configuration)

## Reproducing Paper Results

Follow this complete step-by-step process to reproduce the results from the paper. The workflow consists of:
- **Steps 0-2**: Data preparation and KGE model training (common to both tasks)
- **Step 2a**: Train Link Prediction models (optional, for PPE only)
- **Steps 3-4**: Downstream Task 1 - Pipeline Performance Estimation (PPE)
- **Steps 5-7**: Downstream Task 2 - Dataset Performance Similarity Estimation (DPSE)
- **Step 8**: Ablation study analyzing the impact of MLSeaKG integration on both tasks

Each step builds upon the previous ones.

### Step 0: Generate ExeKGs from OpenML Runs

This step uses the **`openml_exekgs_generation/`** subproject to crawl OpenML and generate ExeKGs from runs and tasks.

**Generate ExeKGs:**
```bash
# Run the ExeKG generation command (from repository root)
python -m openml_exekgs_generation.main --no-mp-runs --mp-tasks --n-runs-per-flow=10
```

**Move Generated Files to the Required Location:**

After generation, move the output ExeKGs and logs to the `kge_experiments/` data directory:
```bash
# Create the target directories if they don't exist
mkdir -p kge_experiments/data/input/datasets/raw/
mkdir -p kge_experiments/data/input/logs/

# Move the generated ExeKGs files to the expected location
mv openml_exekgs_generation/output/exekgs kge_experiments/data/input/datasets/raw/

# Move the tasks log file (required by kge_experiments)
mv openml_exekgs_generation/output/logs/tasks_log.csv kge_experiments/data/input/logs/
```

**Note:** The `tasks_log.csv` file contains metadata about OpenML tasks and datasets that is used by the `kge_experiments` codebase for all subsequent steps.

### Step 1: Prepare Data

Prepare the knowledge graph data by filtering and processing ExeKGs. Performance values are excluded:

```bash
python -m typer kge_experiments.cli.main run prepare-data --filter-by-dataset-ids --excl-flows-per-task --excl-performance-values
```

### Step 2: Train RDF2Vec KGE Model

Train RDF2Vec model.

```bash
python -m typer kge_experiments.cli.main run --use-mlseakg --use-mkga --rdf2vec-d 20 --rdf2vec-w 10 --rdf2vec-ws "random" train-rdf2vec --cpu-count 10 --chunk-size 1000
```

---

## Downstream Task 1: Pipeline Performance Estimation (PPE)

The following steps (2a, 3-4) focus on predicting pipeline performance on datasets. This task evaluates how well embeddings can predict the performance of machine learning pipelines on both seen and unseen datasets.

**Note:** Link Prediction (LP) models are used exclusively for the PPE task and provide an alternative to RDF2Vec embeddings for both dataset and pipeline representations.

---

### Step 2a: Train Link Prediction Models (Optional)

Link Prediction models can be used as an alternative embedding source for pipeline performance prediction. These models are trained on the knowledge graph to learn entity and relation embeddings through the link prediction task.

**Supported Models (all handle literals):**
- `DistMultReaLitEGated`
- `TransEReaLitEGated`
- `DistMultReaLitE`
- `TransEReaLitE`
- `ComplExReaLitEGated`
- `ComplExReaLitE`

**Training command:**
```bash
python -m typer kge_experiments.cli.main run \
  --use-mlseakg --no-use-mkga \
  train-link-prediction \
  --model-type <MODEL> \
  --inverse-triples
```

**Important notes:**
- `--no-use-mkga`: MKGA preprocessing must be disabled for LP models because these models natively handle literals
- `--inverse-triples`: Should always be True (default) for LP models to create bidirectional relationships in the knowledge graph
- `<MODEL>`: Choose from supported models listed above
- Optional: Use `--timeout-hours <HOURS>` to set a training timeout

### Step 3: Pipeline Performance Prediction

The paper evaluates pipeline performance prediction in two scenarios: predicting performance on unseen datasets and predicting performance for unseen pipelines.

#### Step 3.1: Predict Pipeline Performance - Unseen Datasets

```bash
python -m typer kge_experiments.cli.main run \
  --use-mlseakg <--use-mkga|--no-use-mkga> \
  --rdf2vec-d 20 --rdf2vec-w 10 \
  predict-pipeline-performance \
  --split-mode dataset \
  --target <TARGET> \
  --emb-aggr-type concat \
  --model <MODEL> \
  --dataset-emb-source <DATASET_EMB_SOURCE> \
  --pipeline-emb-source <PIPELINE_EMB_SOURCE> \
  --min-train-samples-per-run-id 50 \
  [--pykeen-lp-model-name <LP_MODEL>] \
  [--pykeen-lp-inverse-triples]
```

#### Step 3.2: Predict Pipeline Performance - Unseen Pipelines

```bash
python -m typer kge_experiments.cli.main run \
  --use-mlseakg <--use-mkga|--no-use-mkga> \
  --rdf2vec-d 20 --rdf2vec-w 10 \
  predict-pipeline-performance \
  --split-mode pipeline \
  --target <TARGET> \
  --emb-aggr-type concat \
  --model <MODEL> \
  --dataset-emb-source <DATASET_EMB_SOURCE> \
  --pipeline-emb-source <PIPELINE_EMB_SOURCE> \
  --min-train-samples-per-run-id 1 \
  [--pykeen-lp-model-name <LP_MODEL>] \
  [--pykeen-lp-inverse-triples]
```

**Notes:**
- When `--pipeline-emb-source pykeen-lp` or `--dataset-emb-source pykeen-lp`, you must specify:
  - `--pykeen-lp-model-name`: The LP model name (e.g., `DistMultReaLitEGated`)
  - `--pykeen-lp-inverse-triples`: Must match the training configuration (should be True)
- Use `--no-use-mkga` with PyKEEN LP models (they handle literals natively)
- Use `--use-mkga` with RDF2Vec embeddings
- The paper evaluates all combinations of metafeature types with both RDF2Vec and PyKEEN LP embeddings

### Step 4: Plot Pipeline Performance Prediction Results

Generate comprehensive LaTeX tables combining meta-classification and meta-regression results for pipeline performance prediction.

**Run the script for both split modes:**

```bash
# Generate table for dataset-based splitting (predicting on unseen datasets)
SPLIT_MODE=dataset MIN_TRAIN_SAMPLES=50 python kge_experiments/scripts/plot_pipeline_performance_prediction_results.py

# Generate table for pipeline-based splitting (predicting on unseen pipelines)
SPLIT_MODE=pipeline MIN_TRAIN_SAMPLES=1 python kge_experiments/scripts/plot_pipeline_performance_prediction_results.py
```

**Environment variables:**
- `SPLIT_MODE`: Set to "dataset" or "pipeline" to specify the evaluation scenario
- `MIN_TRAIN_SAMPLES`: Minimum number of training samples per run ID (50 for dataset split, 1 for pipeline split)

**Generated outputs:**

The script generates **two combined LaTeX tables** (one per split mode), each containing both meta-classification and meta-regression results:

1. **Dataset split table**: `combined_results_split_dataset_min_samples_50_table.tex`
   - Columns: Dataset Emb. | Pipeline Strategy | MSE | R² | Accuracy | F1
   - Rows grouped by target metric (accuracy, precision)
   - Shows performance across different dataset embedding sources and pipeline strategies

2. **Pipeline split table**: `combined_results_split_pipeline_min_samples_1_table.tex`
   - Columns: Method | MSE | R² | Accuracy | F1
   - Includes baseline methods (average performance, closest embedding)
   - Rows grouped by target metric (accuracy, precision)

**Table features:**
- **Combined metrics**: Each table includes both meta-regression metrics (MSE, R²) and meta-classification metrics (Accuracy, F1) side-by-side
- **Grouped by target**: Results are organized by target metric rows (e.g., accuracy vs precision)
- **Best performance highlighting**:
  - Bold: Best performance within each embedding source group (dataset split only)
  - Underlined: Overall best performance across all methods
- **Automatic selection**: Shows only the best performing configuration for each embedding source and target combination
- **Baseline comparisons**: Includes metafeature-only baselines alongside RDF2Vec and PyKEEN LP embeddings

---

## Downstream Task 2: Dataset Performance Similarity Estimation (DPSE)

The following steps (5-7) focus on calculating and evaluating dataset similarities using the trained KGE models. This task aims to retrieve similar datasets based on their performance characteristics encoded in the knowledge graphs.

**Note:** Link Prediction models are NOT used for this task. Only RDF2Vec embeddings are used for dataset similarity calculations.

---

### Step 5: Calculate KGE-Based Similarities

Calculate similarities using the trained RDF2Vec models. Run these commands for each configuration used in Step 2:

```bash
python -m typer kge_experiments.cli.main run --use-mlseakg --use-mkga --rdf2vec-d 20 --rdf2vec-w 10 --rdf2vec-ws "random" calculate-kge-similarities --add-data-entity-sim --add-data-entity-and-pipeline-sim --add-pipeline-sim
```

The similarities are calculated in a pairwise fashion for every possible pair of datasets. The results are saved in CSV files for further analysis.

### Step 6: Calculate Graph Edit Distances (GEDs)

Calculate and add Graph Edit Distances to the similarities:

```bash
python -m typer kge_experiments.cli.main run calculate-and-add-geds-to-similarities
```

### Step 7: Plot Dataset Similarity Results

Generate comprehensive plots and LaTeX tables for dataset similarity evaluation results:

```bash
python kge_experiments/scripts/plot_dataset_similarity_results.py --excl_dot_product --excl_manhattan --excl_euclidean --excl_metrics_with_k 10 15 20 0.8
```

**Available arguments for customizing the analysis:**
- `--excl_hit_metrics`: Exclude Hit metrics from the analysis
- `--excl_metrics_with_k`: Exclude specific metrics with k values (e.g., NDCG@5, NDCG@10)
- `--excl_dot_product`: Exclude dot product similarity measurements
- `--excl_manhattan`: Exclude Manhattan distance measurements
- `--excl_euclidean`: Exclude Euclidean distance measurements
- `--only_with_mkga`: Include only results with MKGA preprocessing
- `--only_with_mlsea`: Include only results with MLSeaKG integration

**Generated outputs:**
- Heatmaps comparing baseline methods vs. best KGE approaches
- LaTeX tables for paper publication
- Performance fluctuation plots across different configurations

---

### Step 8: MLSeaKG Ablation Study

Perform a comprehensive ablation study to analyze the impact of MLSeaKG integration on both dataset similarity and pipeline performance prediction tasks.

**For dataset-based splitting (predicting on unseen datasets):**
```bash
MIN_TRAIN_SAMPLES=50 SPLIT_MODE="dataset" python kge_experiments/scripts/mlseakg_ablation_study.py --excl_metrics_with_k 10 15 20 0.8 --excl_dot_product --excl_manhattan --excl_euclidean --only_without_mkga
```

**For pipeline-based splitting (predicting on unseen pipelines):**
```bash
MIN_TRAIN_SAMPLES=1 SPLIT_MODE="pipeline" python kge_experiments/scripts/mlseakg_ablation_study.py --excl_metrics_with_k 10 15 20 0.8 --excl_dot_product --excl_manhattan --excl_euclidean --only_without_mkga
```

**Available arguments:**
- `--excl_metrics_with_k`: Exclude specific metrics with k values (e.g., 10 15 20 0.8)
- `--excl_dot_product`: Exclude dot product similarity measurements
- `--excl_manhattan`: Exclude Manhattan distance measurements
- `--excl_euclidean`: Exclude Euclidean distance measurements
- `--only_without_mkga`: Include only results without MKGA preprocessing

**Environment variables:**
- `SPLIT_MODE`: Set to "dataset" or "pipeline" for pipeline performance prediction analysis
- `MIN_TRAIN_SAMPLES`: Minimum training samples per run ID (50 for dataset split, 1 for pipeline split)

**Generated outputs:**
- **Combined comparison tables**: Side-by-side comparison of results with and without MLSeaKG
  - `mlseakg_ablation_similarity_comparison.tex`: Dataset similarity results
  - `mlseakg_ablation_pipeline_{SPLIT_MODE}_min_samples_{MIN_TRAIN_SAMPLES}_classification_comparison.tex`: Meta-Classification results
  - `mlseakg_ablation_pipeline_{SPLIT_MODE}_min_samples_{MIN_TRAIN_SAMPLES}_regression_comparison.tex`: Meta-Regression results
- **Analysis features**:
  - Identifies and compares best configurations with and without MLSeaKG
  - Generates LaTeX tables
  - Saves results to `kge_experiments/data/ablation_results/` directory

## Citation

If you use this code or our methods in your research, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-032-25156-5_18,
author="Klironomos, Antonis
and Dasoulas, Ioannis
and Periti, Francesco
and Gad-Elrab, Mohamed H.
and Paulheim, Heiko
and Dimou, Anastasia
and Kharlamov, Evgeny",
editor="Acosta, Maribel
and van Erp, Marieke
and Rudolph, Sebastian
and Hartig, Olaf
and Spahiu, Blerina
and Rula, Anisa
and Garijo, Daniel
and Osborne, Francesco",
title="Integrating Meta-features with Knowledge Graph Embeddings for Meta-learning",
booktitle="The Semantic Web",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="336--357",
isbn="978-3-032-25156-5"
}
```

## License

This software is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE.md) file for details.

For a list of open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Acknowledgments

This project includes outsourced code in the following locations:

1. **`MKGA-exekgs-extension/`**: Code adapted from the [MKGA repository](https://gitlab.com/patryk.preisner/mkga).

2. **`kge_experiments/classes/rdf2vec.py`**: Code adapted from [a fork of the pyRDF2Vec repository](https://github.com/MartinBoeckling/pyRDF2Vec/tree/main).
