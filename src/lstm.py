import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from wordcloud import WordCloud, STOPWORDS
from plotly.offline import iplot

lp_train = pd.read_csv("/content/drive/MyDrive/ece542-NN/ABSA/data/Laptop_Train_v2.csv") 
rst_train= pd.read_csv("/content/drive/MyDrive/ece542-NN/ABSA/data/Restaurants_Train_v2.csv")

def data_cleaning(lp_train,rst_train):
  lp_train = lp_train.drop(columns=['from','to','id'])
  rst_train = rst_train.drop(columns=['from','to','id'])
  train_data =pd.concat([rst_train, lp_train])
  train_data.drop(train_data[train_data.polarity == 'conflict'].index, inplace= True)
  train_data['polarity'][train_data['polarity']=='negative'] = 0
  train_data['polarity'][train_data['polarity']=='positive'] = 1
  train_data['polarity'][train_data['polarity']=='neutral']  = 2
  train_data.polarity.value_counts().plot(kind ='bar')
  plot_word_cloud(train_data ,  'Sentence')
  return train_data

stopwords = set(STOPWORDS)
def plot_word_cloud(data ,  column_name):
  corpus = ' '.join(data[column_name])
  corpus = corpus.lower()
  wordcloud = WordCloud(stopwords=stopwords).generate(corpus)
  plt.figure(figsize=(12,12))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()