# torch-rx

torch-rx is pytorch based software for training neural network forcefields for large scale molecular dynamics simmulations. 

the intention of this code is to create pytorch models that can be directly integrated into the RXMD software developed by Ken-ichi Nomura  https://github.com/USCCACS/RXMD


# How to use 
There are two main steps for neural newtork forcefield development :
 1. creation of feature vectors from a given training set that contians given atomic configuration, atomic forces and total energys for the given configs
 2. training of the force-field based from the created feature vectors. 

**All files Required for Training are in /torch-rx/python-training/**

**/torch-rx/miniapp/ contains files for testing reading of trained torch script models in C++**

## Creating Features 

torch-rx currently supports creation of Beheler and Parrinello Radial Style Feature Vectors (https://doi.org/10.1103/PhysRevLett.98.146401) 

- Feature creation can be done serially by running **python create_features.py** and in parrallel with python multiprocessing using **python create_features_multiprocessing.py**

- Training data with positions , forces, and total energy should be in the **.xsf format. (see frc-files for examples)**

- To specify what files to read create a file **fnames** in the same directory as you launch create_features.py

- The format for fnames is the first line is number of files to read and the next lines are the paths to these files.
 
- In **create_features.py** you will supply
     1. Hyper Parameters **eta** (Gaussian Widths) and **RS** (Gaussian centers) for Radial Features
     2. Neural Network Cuttof Distance **RC**
     3. Atomic Energiges for each atom type in the training set 
     4. Max number of atoms for a given configuration in the training set
     
- Created Features will be stored in pytorch binary file TRAINING-DICT.pt  

## Training Model 

- Training is done by running  **python train.py**
- You will need to create a **data** folder for dumped output files
- In **train.py** you will supply 
  1. The pytorch optimizer you wish to use wrapped in **optimizer_function(my_model)** 
  2. An **atomic model** class that defines the model you wish to train. It must have keyword arguments :
      - feature_size : Size of Feature Vector
      - Emax : Max Energy in Training Set
      - Emin : Min Energy in Training Set
      - FeatMax : Max Feature Value in Training set
      - FeatMin : Min Feature Value in Training set
  3. In the main function the following training parameters can be adjusted 
      	    - pF=0.1 #  Force weight in loss
	    - pE=1.0  #  Energy weight in loss
	    - learning_rate=0.0005 # Overides Learning Rate in optimizer
	    - nepochs=200 # Number of epochs to perfrom
	    - per_train=0.95 # percent data set used training 
	    - lstart=True # Restart from previous run
	    - lVerbose=False # Output for train mse every batch iteration 
	    - ldump=True # Dump Predicted Energies and Forces 
	    - lbatch_eval_test=False # Output test mse every batch iteration, coerces lVerbose->True
	    - batch_size=50 # frames in batch
	    - MAX_NATOMS_PER_FRAME=325 # Max atoms in training set +1 
   4. Restarting from previous run can be done by copying dumped model for each atom type from the data folder as TYPE.RESTART.pt 
        - For example if you had an NH3 model you would copy the N and H dumped model you wanted to restart from as N.RESTART.pt and H.RESTART.pt
        
  ## Creating Torch Script Files From Trained Models to run with C++ API
  
  - **convert-and-test.py** is used to convert trained models in torch script files that can be read from C++
  - **convert-and-test.py** will also read from **./fnames-testing** to output predicted energies and forces for a given set xsf files
  
  
  
  
   




 
