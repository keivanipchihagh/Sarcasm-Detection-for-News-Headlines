import json
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('datasets/Sarcasm_Headlines_Dataset.json'))

print('Data Sample:', data[0])


training_size = 20000
vocab_size = 10000
max_len = 100
padding_type = 'post'
truncating_type = 'post'


headlines = []
labels = []

for news in data:
    headlines.append(news['headline'])
    labels.append(news['is_sarcastic'])


training_data = headlines[:training_size]
training_labels = labels[:training_size]

testing_data = headlines[training_size:]
testing_labels = labels[training_size:]


# Create tokenizer object
tokenizer = Tokenizer(num_words = vocab_size, oov_token = '<OOV>')

# Create a word_index dictionary
tokenizer.fit_on_texts(training_data)

# Get word_index dictionary
word_index = tokenizer.word_index

# Tokenize and pad training data
training_sequences = tokenizer.texts_to_sequences(training_data)
training_data_padded = pad_sequences(training_sequences, maxlen = max_len, padding = padding_type, truncating = truncating_type)

# Tokenize and pad testing data
testing_sequences = tokenizer.texts_to_sequences(testing_data)
testing_data_padded = pad_sequences(testing_sequences, maxlen = max_len, padding = padding_type, truncating = truncating_type)

# Sequence sample
print(training_sequences[:3])






