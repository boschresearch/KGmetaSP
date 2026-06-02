# Multimodal Knowledge Graph Augmentation
This code repository is an integral part of a master's thesis focused on evaluating the efficacy of Graph Data Augmentation (GDA) in creating embeddings that are aware of literal information within Knowledge Graphs.

## Introduction
Knowledge Graphs have gained high popularity due to their ability to preserve information and semantic meaning. Here, the task of Knowledge Graph embedding is essential for Knowledge Graph-based Machine Learning and Data Mining tasks. However, embedding approaches seldom consider literal nodes within Knowledge Graphs, focusing solely on entities and their interrelations. This repository aids the underlying thesis in addressing the problem of incorporating literal information into Knowledge Graph embeddings, evaluating existing approaches on this topic, and exploring a data augmentation-based approach.

## Project Structure


```
# Configuration files related to different aspects of the project
├───config
│   ├───aug_approach      # Configuration for augmentation approach
│   ├───aug_method        # Configuration for augmentation method
│   ├───dataload          # Configuration for data loading
│   ├───embed             # Configuration for embedding
│   ├───evaluate          # Configuration for evaluation
│   └───pipeline          # Configuration for pipeline
# Data directories for various stages of data processing
├───data
│   ├───embedded          # Embedded data
│   ├───predicted         # Predicted data
│   ├───preprocessed      # Preprocessed data
│   └───raw               # Raw data
# Documentation files, plots, and tables related to the project
├───docs
│   ├───plots             # Plots and visualizations
│   └───tables            # Tables and documentation
# Jupyter notebooks for analysis and documentation
├───notebooks
# Output directory for project results
├───outputs
# Source code for different project modules
├───src
│   ├───dataload          # Data loading modules
│   ├───embed             # Embedding modules
│   ├───evaluate          # Evaluation modules
│   ├───preprocess        # Data preprocessing modules
│   └───utils             # Utility modules
# Unit tests for project modules
└───tests
```

## Usage
To use the evaluation framework:
- Set up the single.yaml and multiple.yaml files to define the GDA approaches, datasets, embedders, and classifiers for evaluation.
- For a singular evaluation, run main.py located in the src/ folder. This script loads the single.yaml configuration.
- To perform a set of evaluations, similar to those presented in the thesis, run autoevaluate.py in the src/ folder. This script loads and applies the multiple.yaml configuration.
While it is possible to reproduce all test results using the provided seed, we also provide our results along with an evaluation notebook for result inspection. Further the full set of results are available within the all_results.pdf file.
## Installation
*Prerequisites:* Python 3.8
Install kgbench from the repository.
Install the required packages listed in "requirements.txt".

### Libraries Used in This Project
*Data Loading*:
kgbench @ https://github.com/pbloem/kgbench-loader.git
Embedding:
pykeen==1.9.0
pyrdf2vec==0.2.3
Classification:
scikit-learn==1.2.0
GDA Approaches:
pyLDAvis==3.4.0
gensim==4.3.0

*Evaluation and Visualization*:
matplotlib==3.6.3
scipy==1.10.0
seaborn==0.12.2
statannot==0.2.3
statsmodels==0.14.0
tikzplotlib==0.10.1