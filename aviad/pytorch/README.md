# Towards Autoencoding Variational Inference for Aspect-based Opinion Summary
[ARXIV PAPER Towards Autoencoding Variational Inference for Aspect-based Opinion Summary](https://arxiv.org/abs/1902.02507)

Tai Hoang, 
Huy Le, 
and Tho Quan

Applied Artificial Intelligence, 2019; 33(9), 796-816

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Experiments](#experiments)
1. [Topic Coherent Evaluate](#topic-coherent-evaluate)

### Introduction
Autoencoding Variational Inference for Aspect Discovery (AVIAD) model, which extends the previous work of Autoencoding Variational In- ference for Topic Models (AVITM) to embed prior knowledge of seed words. This work includes enhancement of the previous AVI architecture and also modification of the loss function. Ultimately,

### Citation
This code is largely based on [hyqneuron AVITM Pytorch](https://github.com/hyqneuron/pytorch-avitm). If you find the code useful in your research, please cite:

    @article{doi:10.1080/08839514.2019.1630148,
         title={Towards Autoencoding Variational Inference for Aspect-Based Opinion Summary},
         author={Tai Hoang and Huy Le and Tho Quan},
         journal={Applied Artificial Intelligence},
         volume = {33},
         number = {9},
         pages = {796-816},
         year={2019}
         publisher = {Taylor & Francis},
         doi = {10.1080/08839514.2019.1630148},
         URL = {https://doi.org/10.1080/08839514.2019.1630148},
         eprint = {https://doi.org/10.1080/08839514.2019.1630148}
    }

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 18.04.5 LTS)
- Python (We test with Python = 3.6.8 in Anaconda3 = 4.1.1)

### Installation
Download repository:

    $ git clone https://github.com/huylb314/AVIAD_AVIJST.git

Change directory to pytorch aviad version

    $ cd aviad/pytorch

Create anaconda environment `avi`
    
    $ conda env create -f environment.yml

Create anaconda environment `py27`
    
    $ conda env create -f environment_py27.yml

### Experiments
Run the `prodLDA` model in the `URSA` dataset:

    $ cd aviad/pytorch/
    $ source acitvate avi
    $ python run.py

The results will be saved in `results/ursa` with the folder name for each epoch.

### Topic Coherent Evaluate
Compute topic coherent for `URSA` results, we use package implemented by [Jey Han Lau, David Newman and Timothy Baldwin (2014)](https://github.com/jhlau/topic_interpretability.git)

    $ cd aviad/tensorflow/topic_coherent
    $ source acitvate py27
    $ ./clean.sh
    $ ./evaluate_aviad.sh
    
These results will be saved in `oc` folder.

### Contact
- [Tai Hoang](mailto:13520193@gm.uit.edu.vn)
- [Huy Le](mailto:13520360@gm.uit.edu.vn)

### License
See [MIT License](https://github.com/huylb314/AVIAD_AVIJST/blob/master/LICENSE)