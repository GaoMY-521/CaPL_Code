# CPLNet
This is the official repo for our work 'Causality-guided Prompt Learning for Vision-language Models via Visual Granulation', which is accepted by ICCV 2025.

## Setup
We built and ran the repo with CUDA 11.0, Python 3.8.5, and Pytorch 1.13.1. For using this repo, we recommend creating a virtual environment by Anaconda. Please open a terminal in the root of the repo folder for running the following commands and scripts.
```pytorch
conda env create -f requirements.txt
conda activate ldm
```

## Model Training and Evaluation
The CaPL could be trained and evaluated by simply applying the following command:
```pytorch
pyhton3 CaPL.py
```
To use different datasets, users can modify line 338 in ```CaPL.py```.