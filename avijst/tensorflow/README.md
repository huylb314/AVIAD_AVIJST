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

### Introduction
Autoencoding Variational Inference for Aspect-based Joint Sentiment/Topic (AVIJST). Instead of training the JST model using Gibbs sampling, we want to take the advantage ofVariational Autoencoder method which is fast and scalable on large dataset to this joint sentiment/topic model.

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 19.10)
- Python (We test with Python = 3.6.10 in Anaconda3 = 4.8.3)

### Installation
Download repository:

    $ git clone https://github.com/huylb314/AVIAD_AVIJST.git

Change directory to tensorflow aviad version

    $ cd avijst/tensorflow

Create anaconda environment `tf_avi`
    
    $ conda env create -f environment.yml

### Experiments
Run the `AVIJST` model in the `IMDB` dataset, configs are defined in yaml file 

    $ cd avijst/tensorflow/
    $ source activate tf_avi
    $ python run.py --config configs/imdb.yaml

The results will be saved in `results/imdb` with the folder name for each epoch.

### Preprocessing
To preprocess the `IMDB` dataset, configs are defined in yaml file 

    $ cd avijst/tensorflow/
    $ source activate tf_avi
    $ python -c "import nltk; nltk.download('punkt')"
    $ python -c "import nltk; nltk.download('stopwords')"
    $ python preprocess.py --config configs/preprocessing.yaml

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
