# Team MLproj1

## PROJECT 1 : The Process of Discovering the Higgs Boson Particule
The Higgs boson machine-learning challenge aims to explore the potential of advanced machine-learning methods to improve the analysis of data produced by the experiment. This project relies on high dimensional data and targets the creation of a binary classifier that classifies events as an observation of the decay of the Higgs particle into two tau particles.

## Authors
* Asli Andréa : andrea.asli@epfl.ch
* Berquet Romain : romain.berquet@epfl.ch
* Chammas Michel : michel.chammas@epfl.ch
  
## Folders and Files Structure
* **project1_description.pdf :** Project guidelines.
* **project_report :** Written report highlighting the most important findings obtained.

* **data :** provides two files in the .csv format : ***test.csv*** (test set) and ***train.csv*** (training set).

* **scripts :** Provides all the scripts that are needed to implement the project's methods.

    * **run.py :** Runs the algorithms and provides the predictions in the output folder.
    
    * **implementations.py :** Provides all required machine learning methods.
    
    * **proj1_helpers.py :** Provides the helper methods used by the code : _loading the data, loss and gradient computation..._.
    
    * **data_processing.py :** Provides methods for preprocessing the dataset before using any algorithm on it.
    
    * **optimisation.py :** Provides the optimal degree and lambda using k-fold cross-validation (10-fold). 

    * **validation.py :** Provides methods used to execute cross-validation (Data split).

    * **parameters.py :** Contains the parameters we used.
