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
final model.

MEDHA can process a range of tasks which includes 1) Classification 2) Regression (single/multi-target) for Biological sequences and 1) Classification for Proteins. It is equipped with advanced functionalities where a biological sequence can be appended with other numerical features. 

The functionalities of MEDHA are outlined below. 

Data-Processing and scope: This package can directly input either raw sequences to generate one-hot encoded matrices, which it further feeds 
into the CNN based network automatically or  it can directly input PDB-IDs identifying protein-structures, process them (generate graphs then 
datasets ) to  input them in the GNN-based network automatically. 

Automated model construction : This package can create a trained optimal 
neural architecture with optimized hyperparameters. It can then further train the network using K-Fold cross validation to produce the final 
model. This ultimate model from the trained network can then be seamlessly applied to the test dataset (for biological datasets only), with the given framework, saving the 
same for reproducing the results in future.   

Concatenation with handcrafted features: It can additionally concatenate user handcrafted or non-sequential features with the learned vector 
from CNN operation and that can be inputted either to a MLP or a LSTM network. Sequences alone can also be processed with hybrid CNN
LSTM network instead of only CNNs. 

Customizations: For advanced users,  manual adjustments and customizations have been made possible. Instead of building a point and click 
interface,  this package is built as an API like AutoKeras and BioAutoMATED cause this enables further customizations which can be opted by 
advanced users that would  give them the freedom to optimize their models. The complete frameworks which can be applied within 10-15 lines of 
codes ,that are embedded in python scripts so that they can be re-used and used across any operating systems and devices (home 
computers/ARC or HTC clusters) with all python IDEs.  
 
Transfer-Learning for biological sequences: Models achieved for bio-sequence based tasks, can be saved and re-used on independent datasets 
later. 

Automated protein-structure to class classification with  input as PDBs only: Unlike any other available tools, this tool can input PDB-IDs 
identifying protein-structures and with Graphein software embedded within it along with my  custom function, it can directly fetch those PDBs, process them to create protein
graphs and create relevant datasets and apply the same to Graph-architecture-search process (Graph-DARTS) to obtain 
optimal model. This model is then trained with K-Fold cross validation where internally the dataset is divided into 2 parts : train, test . Thus as the end result we obtain  the mean test accuracies and a 
trained model which can be applied on independent test-datasets . 

![image](https://github.com/user-attachments/assets/f47f1c10-b677-428a-922d-49a9cd50c7a8)


Please note that this is a very preliminary work, done in 2023.

