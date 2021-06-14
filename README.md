# Low Resolution Video Action recognition with Resizer Networks. 

## Overview

This repository contains code for running the experiments conducted on the TinyVIRAT dataset. The preprocessed dataset on which the experiments were run can be downloaded from [here](https://drive.google.com/file/d/1ho9kHLapE4WJk9Sodeug7FTm1qWEgiXg/view?usp=sharing). This contains the preprocessed frames as well as the ``classes.txt`` file used in the training experiments. 
The same dataset is also available in the PVC volume ``virat-vr-small`` in the ``amll``  namespace on PRP. 
The experiments involved training I3D, two versions of a resizer network, and addition of a spatial transformer to the resizer network. The networks are available in the following files:
- ``i3d.py`` : I3d
- ``resizer.py`` : The two reszier networks as classes ``ResizerMainNetworkV4_3D`` and ``ResizerMainNetworkV4_2D``
- ``spatial_transformer.py`` : The spatial transforemer network, which was applied sequentially to the network before the resizer. 
- ``spatial_resizer.py`` : The resizer network which combines the transformation within the resizer. This is found in the class ``TransformerWithResizer``. 

The dataset is modelled in the class ``Virat`` present in the file ``virat-dataset.py``. 

## Running code outside PRP:
- The files ``train.py`` , ``test.py`` and ``eval.py`` can be used to run the code outside prp. The usage of the files is as follows: ``train.py <config_file>``. The train test and eval configs can be found in the ``conf`` folder. The evaluation code, which is run locally can by run following the same steps as below for PRP. 

## Running the code on PRP
The steps for setting up PRP can be found at the official prp documentation. (
https://ucsd-prp.gitlab.io/userdocs/start/toc-start/). The dockerfile used to create the image of the container to run the jobs is present in the current repository. Steps followed for the same were part of the prp tutoriaa, using their gitlab instance (https://ucsd-prp.gitlab.io/userdocs/tutorial/images/). The repository used for the same can be found [here](https://gitlab.nautilus.optiputer.net/abhay3010/test-prooject)
The yml files corresponding to the jobs instantiation on PRP can be found in the ``./prp`` folder. The jobs are structred to request the number of GPUs (usually 4) for running the test and evaluation scripts. The trained models are stored in the pvc claim ``virat-vr``. 
All the prp jobs use the ``setup.sh`` script, which takes as input the python script being run. The ``setup.sh`` takes care of updating the github repo on the prp volume and executing the latest version of the code on the container. This was done to avoid creating, pushing and pulling multiple images into the nautilus image registry and save time in spawning jobs. 
## Training:
The ``.yml`` files  that start with training are used to train the model for a particular config. for example , to train the resizer network with i3d, the job ``train_job.yml`` is used. The job can be started using the the command ``kubectl -f train_job.yml``.  When creating a job, modify last value in the ``args`` parameter in the ``yml`` file to reflect the config you wish to train.  The training and validation loss are output on the console of the jobs and can be grepped from the pods using the commands ``kubectl logs -f <pod-name> | grep Loss ``. The output can be locally saved in a file and visualized using the ipython notebook ``eval/Loss Logs.ipynb``. This was done instead of integration with services like tensorboard as they are discouraged for use in prp. The script saves the model weights after every epoch.   
## Testing:
The testing scripts evaluate the f1 macro, f1 micro and accuracy scores of the trained models. They take the model prefix and the range of epochs for which the attributes need to be configured are currently added to the corresponding python script. For example, in order to evaluate the resizer network, the job ``train_job.yml`` can be used. Please make sure that the ``args`` parameter is configured to take the correct config from the ``conf`` folder. 
## Evaluation:
- The model evaluation uses the same same jobs as the ones used in testing with the debug parameter set to True. This results in saving the evaluation files, the default values for which are ``predictions.npy``, ``actuals.npy`` , ``logits.npy`` and ``confusion.npy`` to be saved in the location ``/virat-vr/code/pytorch-i3d/``. The evaluation config can be modified to change the names of these files as required.  For further analysis, these files are copied to the  local machine and analysed using the script ``Analysis.ipynb`` to compute the classwise f1 score for each class. 
- In order to plot the frames corresponding to a given model, the same are copied locally from the prp machines using the command ``kubectl cp prp-container:<path to model> <local path> ``. The same is then used in the function ``sample_resizer_output`` in the function ``eval_resizer.py`` to generate the corresponding frames for a sample of the validation dataset. These can be run locally. 



## Note
Thie repository was forked from [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) and additional models were added to it as part of the project. 
