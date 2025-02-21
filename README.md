# MEDHA_AutoML_for_Biological_data
 Preliminary work in DPhil - Automated Deep Learning Framework  for Biological sequences and structures
This library, focuses on bridging the gap between molecular biology and deep learning through the development of MEDHA, an end-to-end automated deep learning pipeline for biological sequence and structure analysis. This tool, which I have built independently, automates the entire modelling workflow from data pre-processing (including protein graph generations and biological sequences processing) through model generation, optimization, training, and benchmarking. By leveraging neural architecture search (NAS) algorithms with novel search space designs for both CNNs and GNNs, alongside hyperparameter optimization (HPO), MEDHA efficiently identifies high-performing architectures for analysing DNA, RNA, protein sequences, and complex biological structures (proteins).  

This is a prelim work. Recent work are in progress and will be uploaded in Github once subitted to journals/conferences.

![image](https://github.com/user-attachments/assets/78de681d-e25f-464b-a352-98ea901d328a)

 A very high-level outline of MEDHA. The system 
accepts both biological sequences and protein structures in its 
framework and pre-processes them accordingly. The processed 
data (one-hot encoded matrix/graphs) are then fed into the next 
module which constitutes DARTS, HyperOPT and K-Fold CV. 
On a high level, DARTS and HyperOPT design a model (sub
optimal) which is further trained with K-Fold CV to make the 
final model

