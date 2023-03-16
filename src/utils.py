
def data_cleaning(lp_train,rst_train):
  lp_train = lp_train.drop(columns=['from','to','id'])
  rst_train = rst_train.drop(columns=['from','to','id'])
  train_data =pd.concat([rst_train, lp_train])
  train_data.drop(train_data[train_data.polarity == 'conflict'].index, inplace= True)
  train_data['polarity'][train_data['polarity']=='negative'] = 0
  train_data['polarity'][train_data['polarity']=='positive'] = 1
  train_data['polarity'][train_data['polarity']=='neutral']  = 2
  train_data.polarity.value_counts().plot(kind ='bar')
  #plot_word_cloud(train_data ,  'Sentence')
  train_data["polarity"] = pd.to_numeric(train_data["polarity"],errors='coerce')
  X_train, X_test, y_train, y_test = train_test_split(train_data,train_data['polarity'],stratify=train_data['polarity'], test_size=0.2, random_state=1)
  return X_train, X_test, y_train, y_test

def plot_word_cloud(data ,  column_name):
  corpus = ' '.join(data[column_name])
  corpus = corpus.lower()
  wordcloud = WordCloud(stopwords=stopwords).generate(corpus)
  plt.figure(figsize=(12,12))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()

def tokenization(X_train, x_val,Y_train, y_val):
  tokenizer.fit_on_texts(list(X_train.Sentence))
  text_X_train_tokenized  = tokenizer.texts_to_sequences(X_train.Sentence)
  text_x_val_tokenized    = tokenizer.texts_to_sequences(x_val.Sentence)
  aspect_X_train_tokenized= tokenizer.texts_to_sequences(X_train['Aspect Term'])
  aspect_x_val_tokenized  = tokenizer.texts_to_sequences(x_val['Aspect Term'])
  max_len = max( len(tokenized) for tokenized in text_X_train_tokenized)
  return text_X_train_tokenized, text_x_val_tokenized, aspect_X_train_tokenized, aspect_x_val_tokenized, max_len

def word_embeddings(text_X_train_tokenized, text_x_val_tokenized, aspect_X_train_tokenized, aspect_x_val_tokenized, max_len):
  x_train_text_pad  = pad_sequences(text_X_train_tokenized, maxlen=max_len)
  x_train_aspect_pad= pad_sequences(aspect_X_train_tokenized, maxlen=1)
  x_val_text_pad    = pad_sequences(text_x_val_tokenized, maxlen=max_len)
  x_val_aspect_pad  = pad_sequences(aspect_x_val_tokenized, maxlen=1)

  x_train = [x_train_text_pad, x_train_aspect_pad]
  x_vald  = [x_val_text_pad,x_val_aspect_pad]

  return x_train, x_vald 

def lstm(lstm_units =512 , max_len=1):
  input_text = Input(shape = (max_len,))
  input_aspect = Input(shape = (1,),)

  word_embedding  = Embedding(NUM_WORDS , EMBEDDING_SIZE, input_length= max_len)
  text_embedding  = SpatialDropout1D(0.2)(word_embedding(input_text))

  asp_embedding   = Embedding(NUM_WORDS, EMBEDDING_SIZE , input_length=max_len)
  aspect_embedding= asp_embedding(input_aspect)

  aspect_embedding = Flatten()(aspect_embedding)
  rep_aspect_embd  = RepeatVector(max_len)(aspect_embedding)

  input_concat = concatenate([text_embedding, rep_aspect_embd], axis = -1)
  hidden = LSTM(lstm_units)(input_concat) 
  Dense_layer = Dense(128, activation ='relu')(hidden)
  output_layer= Dense(3, activation = 'softmax')(Dense_layer)
  return Model([input_text, input_aspect], output_layer)

def ae_lstm(lstm_units = 512, max_len=1 ):
    input_text = Input(shape=(max_len,))
    input_aspect = Input(shape=(1,),)
    
    word_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=max_len)
    text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))
    
    asp_embedding = Embedding(NUM_WORDS, EMBEDDING_SIZE, input_length=max_len)
    aspect_embed = asp_embedding(input_aspect)
    
    
    aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
    repeat_aspect = RepeatVector(max_len)(aspect_embed)  # repeat aspect for every word in sequence

    input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
    hidden = LSTM(lstm_units)(input_concat)
    Dense_layer  = Dense(128, activation='relu')(hidden)
    output_layer = Dense(3, activation='softmax')(Dense_layer)
    return Model([input_text, input_aspect], output_layer)