MEDHA: AutoML for Biological Data (VERSION 0.1)

 Preliminary work in DPhil - Automated Deep Learning Framework  for Biological sequences and structures (done in 2023)

 
MEDHA (sanskrit : Intellect) is an end-to-end automated deep learning pipeline for biological sequence and structure analysis. This framework bridges the gap between molecular biology and deep learning by automating the entire modeling workflow from data pre-processing to model generation through neural-architecture-search, model optimization by hyperparameter-optimization , model training by k-fold cross-validation to  model deployment.

This is a prelim work. Recent work are in progress and will be uploaded in Github once subitted to journals/conferences.

![image](https://github.com/user-attachments/assets/78de681d-e25f-464b-a352-98ea901d328a)

 A very high-level outline of MEDHA. The system 
accepts both biological sequences and protein structures in its 
framework and pre-processes them accordingly. The processed 
data (one-hot encoded matrix/graphs) are then fed into the next 
module which constitutes DARTS, HyperOPT and K-Fold CV. 
On a high level, DARTS and HyperOPT design a model (sub
optimal) which is further trained with K-Fold CV to make the 
final model.

Key Features

Versatile Inputs: Processes both biological sequences (DNA/RNA/protein) and protein structures

Comprehensive Task Support:

Sequences: Classification and regression (single/multi-target)
Proteins: Structure-based classification

Automated Architecture Design: (neural architecture search)

Hyperparameter Optimization:(hyperparameter optimization)

Robust Validation: (K-FOLD CV)

Feature Integration: Can combine sequence data with handcrafted features

Advanced Graph Processing: Custom protein graph generation using Graphein and custom functions



Workflow

Data Processing: Generates one-hot encoded matrices from sequences or protein graphs from PDB IDs

Architecture Search: Applies DARTS (https://arxiv.org/abs/1806.09055) with novel search-space to identify optimal neural architectures (for both CNN, GNN)

Hyperparameter Optimization: Uses HyperOPT (https://hyperopt.github.io/hyperopt/)  to find optimal parameters

Model Training: Implements K-Fold CV for robust performance

Model Deployment: Provides trained models ready for application to new data 

Additional information Regarding Templates

![image](https://github.com/user-attachments/assets/f47f1c10-b677-428a-922d-49a9cd50c7a8)

How to run this package :
The framework is designed for simple integration with your workflow:

# Clone the repository
git clone https://github.com/shrutiOx/MEDHA_AutoML_for_Biological_data.git

# Install dependencies
pip install -r DEPENDENCIES.txt

# Usage instructions:
# 1. Copy the MEDHA folder (main framework) to your desired template folder within MEDHA_Frameworks
# 2. Modify the template file parameters according to your dataset
# 3. Run the template file

# Example:
python MEDHA_Frameworks/Classification_Advanced_AutoHPODART/Classification_Advanced_AutoHPODART.py

Note: The protein structure processing functionality is currently available in the sequence-based modules only. Complete model reproduction for protein structures is under development.
Status
This repository contains preliminary work from 2023. More recent developments will be published in upcoming papers and shared in GitHub as next versions.
Dependencies
All required dependencies are listed in DEPENDENCIES.txt file in the repository.




