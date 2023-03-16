NUM_WORDS = 100000
EMBEDDING_SIZE = 128
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, )