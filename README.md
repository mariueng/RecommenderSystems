# TDT4215 Group Project Spring 2022

This repository contains the code for group 10 for the course TDT4215 Recommender Systems at IDI, NTNU.

To git started, please setup your local environment following the instructions below and go to the **main.ipynb** notebook to run the models.

NB! Data used in the models have been preprocessed and hence making it difficult to automatically run the models from the original dataset. To download the preprocessed datasets, please visit [this drive][https://drive.google.com/drive/folders/1osf88CZsjEeatSWAjds0xZShTG4HZNwC?usp=sharing].

## Setup

To be able to use this project you need an environment supporting installment of the necessary python packages and jupyter notebook (kernels).
This can be done using pip, pipenv, conda, and jupyter notebook.

### Setup using Pipenv

To get started using pipenv you just have to install pipenv and run

`pipenv shell`

followeb by

`pipenv install`

Ensure the correct interpreter is selected in your IDE and you should be ready to go.

NB! You might have to set a kernel for the jupyter notebooks as well manually.

### Other packages

To be able to run the content-based article recommender model you first need to download an NLTK package. After installing the nltk package in your environment, start a python environment and type the following:

`nltk.download("stopwords")`

See [nltk.org](https://www.nltk.org/) for more information.
