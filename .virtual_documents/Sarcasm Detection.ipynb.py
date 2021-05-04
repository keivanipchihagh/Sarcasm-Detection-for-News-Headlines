import json
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


with open("Datasets/sarcasm.json", 'r') as f:
    dataset = json.load(f)

print('Data Sample:', data[0])


training_size = 20000
vocab_size = 10000
max_len = 100
padding_type = 'post'
truncating_type = 'post'
embedding_dim = 16


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

# Convert lists to tensors
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


# Sequence sample
print(training_sequences[:3])


# Define model structure
model = keras.models.Sequential()

# Define Layers
model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_len))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(units = 24, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Compile Model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Model summary
model.summary()


epochs = 30
history = model.fit(x = training_data_padded, y = training_labels, epochs = epochs, validation_data = (testing_data_padded, testing_labels))


def plot(history, metric):
    plt.plot([i for i in range(epochs)], history.history[metric], label = metric)
    plt.plot([i for i in range(epochs)], history.history['val_' + metric], label = 'Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    
plot(history, 'loss')
plot(history, 'accuracy')


sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]

# Tokenize and pad sentences
sequences = tokenizer.texts_to_sequences(sentence)
sentences_padded = pad_sequences(sequences, maxlen = max_len, padding = padding_type, truncating = truncating_type)

# Predict
print(model.predict(sentences_padded))









