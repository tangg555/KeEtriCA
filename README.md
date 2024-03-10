# KeEtriCA-storygeneration
This repository is the code and resources for the paper [A Cross-Attention Augmented Model for Event-Triggered Context-Aware Story Generation](https://arxiv.org/abs/2311.11271) 

## Introduction
This project is implemented with **Pytorch**.

This project is implemented based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, a framework to ease the training of pytorch. If you dislike it, no worry, you can copy the model files (in `src/models`) and datasets files (in `src/modules`) to your own repository, and train them with your code.

All the pretrained model used here are downloaded from [Huggingface](https://huggingface.co/docs). E.g. [BART](https://aclanthology.org/2020.acl-main.703.pdf) is downloaded from [Hugginface: bart-base](https://huggingface.co/facebook/bart-base).

If you are a freshman in NLG area and feel hard to read the code, I prepared a story generation demo for you ([demo](https://github.com/tangg555/story-generation-demo)).

## Prerequisites
The key dependencies I have for running the code:
- Python 3 or Anaconda (mine is v3.8)
- [Pytorch](https://pytorch.org/) (mine is v1.13.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) v4.27.4
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) v1.9.5
- all the packages listed in the file `requirements.txt`

## Quick Start

### 1. Install packages
Install the aforementioned prerequisites, and run
```shell
pip install -r requirements.txt
```

### 2. Collect Datasets and Resources
`datasets` and `resources` are not included in the code, since their sizes are too large. 
Both of them can be downloaded from [datasets](https://www.dropbox.com/s/b007zce28ou52va/datasets.zip?dl=0)
and [resources](https://www.dropbox.com/s/wr9sxhhu4qteq2t/resources.zip?dl=0) . 
Unzip it at the base directory.

If you intend to preprocess the data by yourself, please read following instructions. Otherwise, please skip to the next section.

The **raw dataset** we suggest download

**Preprocess**

Put your downloaded raw dataset (we downloaded it from [HINT](https://github.com/thu-coai/HINT)) to `resources/raw_data`. 

Take roc-stories for an example, you need make it be `resources/raw_data/thu-coai-hint/roc-stories`.

Run `preprocessing/event_trigger/hint_roc_stories_helper.py`, and then `preprocessing/event_trigger/event_annotator.py`, and you will have `resources/datasets/event-plan/roc-stories`.

Similarly, if you want to run HINT as a story generation model for experiments, you need to download HINT dataset from [HINT](https://github.com/thu-coai/HINT), and make it to be `/datasets/thu-coai-hint/roc-stories`.

`preprocessing/event_trigger/hint_roc_stories_helper.py`
`preprocessing/event_trigger/hint_writing_stories_helper.py`
`preprocessing/event_trigger/hint_book_corpus_helper.py`
are the preprocessing codes for the three data corpus.


### 3. Training and Tesing
Please read the codes in `tasks`, and you will understand how it works.

Some commands for training and testing refer to the file `python_commands.sh`

## Citation
If you found this repository or paper is helpful to you, please cite our paper. 

This is the arxiv citation:
```angular2
@article{tang2023cross,
  title={A Cross-Attention Augmented Model for Event-Triggered Context-Aware Story Generation},
  author={Tang, Chen and Loakman, Tyler and Lin, Chenghua},
  journal={arXiv preprint arXiv:2311.11271},
  year={2023}
}
```

