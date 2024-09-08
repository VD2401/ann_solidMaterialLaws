# ANN_SolidMaterialLaws
This project aims at simulating the constitutive laws of solid materials using Artificial Neural Networks.

The model chosen for test is a 3D U-Net. It has previously demonstrated its effectiveness on the 2D problem.

The code has not been optimized as the project was mainly results oriented.
It is organized the following way:

The `dataset.py` file contains a Dataset class which allows to load the datafiles with a specific format of 128 samples.

The `augmentation.py` provide functions for augmenting the dataset.

The `model.py` files presents the code for the U-Network. Parameters of the model can be modified at initialisation to test different configurations. 

The `training.py` file contains the architecture for the training with train/test loops. It contains the different hyperparameters. 

Visualization.py contains two functions. One for plotting the volume. The second for plotting 4 2D slices of a block: input, target, output, error (ITOE).

The different files starting by `main` are the file executed for testing the different parameters. 

The notebooks can help for visualization of results.

We use the *mlflow* library for recording the runs and help compare them together. With the code implemented in the `main` file, parameters are automatically recorded at run time in the mlflow folder. At each node of the mlflow folder tree there is a meta.yaml file that provides information. 
Each "Experiment" is a test for a parameter (can be multiple but here one). An experiment has an ID (list of numbers) and a name. The folder has the name of the ID. 
Each "Run" tests a value for a parameter (or multiple). Runs have a name (can be shared between runs) but have a unique ID (sequence of letters and numbers).  
To visualize the results: Install a python environment with pip or conda with mlflow library. In the terminal, activate the environment. Cd to the folder containing the `mlruns` folder. Finally, enter `mlflow ui`.
