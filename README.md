![GitHub](https://img.shields.io/github/license/sushanthreddyC/Aspect-Based-Sentiment-Analysis?color=brightgreen&style=flat) <br/>
![GitHub language count](https://img.shields.io/github/languages/count/sushanthreddyC/Aspect-Based-Sentiment-Analysis?style=flat) ![GitHub repo file count](https://img.shields.io/github/directory-file-count/sushanthreddyC/Aspect-Based-Sentiment-Analysis?style=flat) ![GitHub repo size](https://img.shields.io/github/repo-size/sushanthreddyC/Aspect-Based-Sentiment-Analysis?style=flat) <br/>
![GitHub contributors](https://img.shields.io/github/contributors/sushanthreddyC/Aspect-Based-Sentiment-Analysis?color=brightgreen&style=flat) ![GitHub last commit](https://img.shields.io/github/last-commit/sushanthreddyC/Aspect-Based-Sentiment-Analysis?style=flat)

# Aspect-Based-Sentiment-Analysis

This repository contains the code and resources for Aspect-Based Sentiment Analysis (ABSA). ABSA is a natural language processing (NLP) task that aims to identify and analyze sentiments expressed towards specific aspects or entities in text.

## Overview

The objective of this project is to develop a model that can accurately predict sentiment towards different aspects within a given text and the given aspect. It predicts the sentiment associated with each aspect, categorizing it as positive, negative, or neutral.

## Dataset Used
The data is taken from kaggle and is available ![here](https://github.com/sushanthreddyC/Aspect-Based-Sentiment-Analysis/tree/main/data). The files named Laptop_Train_v2 and Restraunts_Train_v2 are used for training and validation of the models.

## Models Trained
The models and architecture used for ABSA are:

### Bi-LSTM
Architecture:


![image](https://github.com/psvkaushik/Aspect-Based-Sentiment-Analysis/assets/86014345/ec59fc82-fd4d-4885-ab7b-133769ff0422)

The Results observed:


![image](https://github.com/psvkaushik/Aspect-Based-Sentiment-Analysis/assets/86014345/29d773bf-f7d6-43c2-bade-7644d3528369)

### Attention Based Model with Aspect Embedding

This model is based on the attention mechanism, which allows the model to focus on relevant parts of the text while making predictions. This approach helps to capture the sentiment expressed towards specific aspects more effectively. The ![MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention) module from tensorflow was used as the attention layer. The aspect word is also embedded into the input

Architecture:

![image](https://github.com/psvkaushik/Aspect-Based-Sentiment-Analysis/assets/86014345/68a8c8c7-9fa0-42e8-bc2e-ebc52d8ed57e)


Results Observed:

![image](https://github.com/psvkaushik/Aspect-Based-Sentiment-Analysis/assets/86014345/5fe33ed7-9465-4203-bb42-c10a13ae74ac)

The code for the above two models are found ![here](https://github.com/psvkaushik/Aspect-Based-Sentiment-Analysis/tree/main/Notebooks)

### Fine-Tuning BERT

_Coming soon....._

