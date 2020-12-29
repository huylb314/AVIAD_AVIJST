# Towards Autoencoding Variational Inference for Aspect-based Opinion Summary
[ARXIV PAPER Towards Autoencoding Variational Inference for Aspect-based Opinion Summary](https://arxiv.org/abs/1902.02507)

Tai Hoang, 
Huy Le, 
and Tho Quan

Applied Artificial Intelligence, 2019; 33(9), 796-816

### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Autoencoding Variational Inference for Aspect Discovery (AVIAD)](#AVIAD)
1. [Autoencoding Variational Inference for Aspect-based Joint Sentiment/Topic (AVIJST)](#AVIJST)
1. [Experiments](#experiments)
1. [Topic Coherent Evaluate](#topic-coherent-evaluate)
1. [Citation](#citation)

### Introduction
Autoencoding Variational Inference for Aspect Discovery (AVIAD) model, which extends the previous work of Autoencoding Variational In- ference for Topic Models (AVITM) to embed prior knowledge of seed words. This work includes enhancement of the previous AVI architecture and also modification of the loss function.

Autoencoding Variational Inference for Aspect-based Joint Sentiment/Topic (AVIJST). Instead of training the JST model using Gibbs sampling, we want to take the advantage ofVariational Autoencoder method which is fast and scalable on large dataset to this joint sentiment/topic model.

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 19.10)
- Python (We test with Python = 3.6.10 in Anaconda3 = 4.8.3)

### AVIAD tensorflow
Run the `AVIAD` model in the `URSA` dataset, configs are defined in yaml file 

    $ cd aviad/tensorflow

### AVIAD pytorch
Run the `AVIAD` model in the `URSA` dataset, configs are defined in yaml file 

    $ cd aviad/pytorch

### AVIJST tensorflow
Run the `AVIJST` model in the `IMDB` dataset, configs are defined in yaml file 

    $ cd avijst/tensorflow

### AVIJST pytorch
Run the `AVIJST` model in the `IMDB` dataset, configs are defined in yaml file 

    $ cd avijst/pytorch

### Contact
- [Huy Le](mailto:13520360@gm.uit.edu.vn)

### Citation
If you find the code useful in your research, please cite:

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

### License
See [MIT License](https://github.com/huylb314/AVIAD_AVIJST/blob/master/LICENSE)
