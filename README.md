This code was used for Text Mining course project

The `logiqa` folder contains code for testing models on the LogiQA dataset

The `squad_newsqa` folder contains code for testing models on the SQuAD and the NewsQA dataset.

## Setup
1. To run this code you first should create environment:
    ```bash
    conda env create -f environment.yml
    ```
    1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
2. Then you need to activate the environment (cd into folder where the project is located in)
    ```bash
    source activate logiqa
    ```
    or
    ```bash
    source activate squad
    ```
3. Run `python setup.py`
4. Finally, `python train.py -h`
    1. You may find it helpful to browse the arguments provided by the starter code.

_Note: Due to the large size of the NewsQA and SQuAD datasets, I am unable to upload them to this repository. If you require access to these datasets, please don't hesitate to contact me and I will be happy to provide them to you._