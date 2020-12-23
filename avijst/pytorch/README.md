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

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 19.10)
- Python (We test with Python = 3.6.10 in Anaconda3 = 4.8.3)

### Installation
Download repository:

    $ git clone https://github.com/huylb314/AVIAD_AVIJST.git

Change directory to tensorflow aviad version

    $ cd aviad/tensorflow

Create anaconda environment `tf_aviad`
    
    $ conda env create -f environment.yml

### Experiments
Run the `prodLDA` model in the `URSA` dataset, configs are defined in yaml file 

    $ cd aviad/tensorflow/
    $ source acitvate tf_aviad
    $ python run.py --config configs/1k.yaml

The results will be saved in `results/ursa` with the folder name for each epoch.

### Topic Coherent Evaluate
Compute topic coherent for `URSA` results, we use package implemented by [Jey Han Lau, David Newman and Timothy Baldwin (2014)](https://github.com/jhlau/topic_interpretability.git)

    $ cd aviad/tensorflow/topic_coherent
    $ bash clean.sh
    $ bash evaluate_aviad.sh -f ../results/ursa/1k/ -c ../corpus/ursa/ -n 3 -t 50

```
evaluate_aviad.sh
-f : topwords folder
-c : corpus folder
-n : number of topics
-t : number of coherent topwords
```

```
Results:
    oc folder: result file.
    final_log.txt: summary scores.
```

### Preprocessing
To preprocess the `URSA` dataset, configs are defined in yaml file 

    $ cd aviad/tensorflow/
    $ source acitvate tf_aviad
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
