# Low Resolution Video Action recognition with Resizer Networks. 

## Overview

This repository contains code for running the experiments conducted on the TinyVIRAT dataset. The preprocessed dataset on which the experiments were run can be downloaded from [here](https://drive.google.com/file/d/1ho9kHLapE4WJk9Sodeug7FTm1qWEgiXg/view?usp=sharing). This contains the preprocessed frames as well as the ``classes.txt`` file used in the training experiments. 
The same dataset is also available in the PVC volume ``virat-vr-small`` in the ``amll``  namespace on PRP. 
The experiments involved training I3D, two versions of a resizer network, and addition of a spatial transformer to the resizer network. The networks are available in the following files:
- ``i3d.py`` : I3d
- ``resizer.py`` : The two reszier networks as classes ``ResizerMainNetworkV4_3D`` and ``ResizerMainNetworkV4_2D``
- ``spatial_transformer.py`` : The spatial transfoemer network, which was applied sequentially to the network before the resizer. 
- ``spatial_resizer.py`` : The resizer network which combines the transformation within the resizer.(This file is still being modified)

The dataset is modelled in the class ``Virat`` present in the file ``virat-dataset.py``. 

## Running the code on PRP
The steps for setting up PRP can be found at the official prp documentation. (
https://ucsd-prp.gitlab.io/userdocs/start/toc-start/)
The yml files corresponding to the jobs instantiation on PRP can be found in the ``./prp`` folder. The jobs are structred to request the number of GPUs (usually 4) for running the test and evaluation scripts. The trained models are stored in the pvc claim ``virat-vr``. 
All the prp jobs use the ``setup.sh`` script, which takes as input the python script being run. The ``setup.sh`` takes care of updating the github repo on the prp volume and executing the latest version of the code on the container. This was done to avoid creating, pushing and pulling multiple images into the nautilus image registry and save time in spawning jobs. 
## Training:
The ``.yml`` files  that start with training are used to train the model for a particular config. for example , to train the resizer network with i3d, the job ``train_resizer.yml`` is used. The job can be started using the the command ``kubectl -f train_resizer.yml``.  The training and validation loss are output on the console of the jobs and can be grepped from the pods using the commands ``kubectl logs -f <pod-name> | grep Loss ``. This was done instead of integration with services like tensorboard as they are discouraged for use in prp. The script saves the model weights after every epoch.   
## Testing:
The testing scripts evaluate the f1 macro, f1 micro and accuracy scores of the trained models. They take the model prefix and the range of epochs for which the attributes need to be configured are currently added to the corresponding python script. For example, in order to evaluate the resizer network, the job ``job_eval_resizer.yml`` can be used. The script being called in the job is where the parameters need to be configured (``eval_resizer_spatial.py``) . 

## Evaluation:
- The model evaluation uses the same same jobs as the ones used in testing with the debug parameter set to True. This results in saving the evaluation files, ``predictions.npy``, ``actuals.npy`` , ``logits.npy`` and ``confusion.npy`` to be saved in the location ``/virat-vr/code/pytorch-i3d/``. The same are copied to the local machine and analysed using the script ``Analysis.ipynb`` to compute the classwise f1 score for each class. 
- In order to plot the frames corresponding to the 





## Note
Thie repository was forked from [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) and additional models were added to it as part of the project. 
