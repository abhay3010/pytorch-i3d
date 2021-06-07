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

##Running the code on PRP
The yml files corresponding to the jobs instantiation on PRP can be found in the ``./prp`` folder. The jobs are structred to request the number of GPUs (usually 4) for running the test and evaluation scripts. The trained models are stored in the pvc claim 




## Note
Thie repository was forked from [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) and additional models were added to it as part of the project. 
