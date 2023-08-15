# Topic Classification using DistilBERT for Banking77

## Introduction
This repository contains the implementation of **Topic Classification using DistilBERT for Banking77**, wherein we implement a [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert).

## Data Generation
I have used the [Banking77](https://huggingface.co/datasets/banking77) dataset with 77 topic classes. Following are certain data statistics:

### Dataset Statistics
| **No. of Train Instances** | **No. of Test Instances** | **No. of Classes** |
|--------------------------  |---------------------------|--------------------|
| 10003                      | 3080                      | 77                 |

## Results
Test Accuracy = **89.01%**

## Directory Structure
The `src` folder contains the source code.
```
    .
    ├── ...
    ├── environment.yml  : Create conda env with modules.
    ├── requirements.txt : Install modules using pip after env creation.
    ├── model            : Contains the saved model and tokenizers.
    ├── src                    
    │   ├── app.py       : The main script to run the Gradio UI.
    |   ├── main.py      : Train and save model and tokens.
    |   ├── utils        : Contains utility files.
    └── ...
```

## Installation Steps
### Prerequisites
You'll need to have [`git`](https://git-scm.com/). Check using:
```
git version
```

### Cloning the repository
Clone the repo using the following:
```
git clone https://github.com/vinamra-baghel/topic-classification-banking77.git
cd 'topic-classification-banking77'
```

### Environment Setup and Installing Packages
Create a conda environment:

```
conda create --name topicbank77 python=3.11
```

After creating the environment, activate the environment using the following command:

```
conda activate topicbank77
```

Navigate to the `src` folder by using the command:

```
cd src
```

Install the dependencies required for this project using the following command:

```
pip install -r requirements.txt
```

Alternatively, create a conda environment directly using the environment file: 

```
conda env create -f environment.yml
```

## How to run?
Download the model files and save them in the `model` directory from [here](https://drive.google.com/drive/folders/1L5afuFKXpSKqLACwHNIP-BXaOeARWLC9?usp=sharing). Then, in the `src` directory, simply run `app.py` to run the Gradio UI. 

To train and save the model, run `main.py`.

## Citation
```
@inproceedings{Casanueva2020,
    author      = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
    title       = {Efficient Intent Detection with Dual Sentence Encoders},
    year        = {2020},
    month       = {mar},
    note        = {Data available at https://github.com/PolyAI-LDN/task-specific-datasets},
    url         = {https://arxiv.org/abs/2003.04807},
    booktitle   = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020}
}
```