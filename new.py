import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from tensorflow.python.keras.models import load_model
import keras.utils as  ks
from word2vec import word2vec
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, regularizers

a=word2vec()
data=pd.read_csv('new_symptoms_vs_specialists.csv',usecols=['complaint','specialist','system'])


data = data.applymap(lambda s:s.lower().strip() if type(s) == str else s)
data['complaint'].dropna(inplace=True)
data['specialist'].dropna(inplace=True)
data.dropna(subset=['complaint','specialist'],inplace=True)

new_list=[]
for i in data['specialist']:
    new_list.append(i)
unique_specialists=set(list(new_list))
list_new=list(unique_specialists)

new_dict={}
for i in range(len(unique_specialists)):
    new_dict[list_new[i]] = i
#print(new_dict)

for i in data['specialist']:
     if i in new_dict.keys():
         data.loc[data['specialist']== i,'labels'] = new_dict[i]
labels = ks.to_categorical(data['labels'], num_classes=len(unique_specialists))

n_most_common_words = 8000
max_len = 200
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['complaint'].values)
sequences = tokenizer.texts_to_sequences(data['complaint'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)
epochs = 100
emb_dim = 128
batch_size = 128
print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))


if(os.path.exists("models/myModel.h5") == False):
    model = Sequential()
    e = Embedding(30000, 300, weights=[a.embeddings], input_length=X.shape[1])
    e.trainable = False
    model.add(e)
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(256, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(len(unique_specialists), activation='softmax',kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2)

    model.save("models/myModel.h5")
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

txt = ["I have  pain in my knees"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len)
loaded_model=load_model("models/myModel.h5")

pred = loaded_model.predict(padded)
accr = loaded_model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
print(pred, list_new[np.argmax(pred)])


# new_list_x=[]
# new_list_y=[]
#
# for i in data['complaint']:
#     new_list_x.append(i)
#
# for i in data['specialist']:
#     new_list_y.append(i.strip().lower())
#
# num_unique_specialists=set(list(new_list_y))
# new_dict={}
#
# new=list(set(num_unique_specialists))
#
# for i in range(len(new)):
#     new_dict[(new[i])] = i
# print(new_dict)
#
# new_list=list(new_dict.values())
# print(new_list)
#
#
# def clean_text(txt):
#     txt = "".join(v for v in txt if v not in string.punctuation).lower()
#     txt = txt.encode("utf8").decode("ascii",'ignore')
#     return txt
#
# corpus = [clean_text(x) for x in new_list_x]
#
# tokenizer = Tokenizer()

#for i in data['specialist']:
#     if i.strip().lower() in new_dict.keys():
#         data['label'][i] = new_dict[i.strip().lower()]
# labels = ks.to_categorical(data['label'], num_classes=21)
# print(labels[:4])

# def get_sequence_of_tokens(corpus):
#     ## tokenization
#     tokenizer.fit_on_texts(corpus)
#     total_words = len(tokenizer.word_index) + 1
#
#     ## convert data to sequence of tokens
#     input_sequences = []
#     for line in corpus:
#         token_list = tokenizer.texts_to_sequences([line])[0]
#         for i in range(1, len(token_list)):
#             n_gram_sequence = token_list[:i + 1]
#             input_sequences.append(n_gram_sequence)
#     return input_sequences, total_words
#
#
# def generate_padded_sequences(input_sequences):
#     max_sequence_len = max([len(x) for x in input_sequences])
#     input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
#
#     predictors, label = input_sequences,new_list_y
#     labels = ks.to_categorical(new_list,len(num_unique_specialists))
#     return predictors, labels, max_sequence_len
#
#
# inp_sequences, total_words = get_sequence_of_tokens(corpus)
# predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
# print(max_sequence_len)
# print(total_words)
#
# def create_model(max_sequence_len, total_words):
#     input_len = max_sequence_len
#     model = Sequential()
#
#     # Add Input Embedding Layer
#     model.add(Embedding(total_words, 10, input_length=input_len))
#
#     # Add Hidden Layer 1 - LSTM Layer
#     model.add(LSTM(100))
#
#     # Add Output Layer
#     model.add(Dense(21, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#
#     return model
#
#
# model = create_model(max_sequence_len, total_words)
# print(model.summary())
# model.fit(predictors, label, epochs=100, verbose=2)
#
#
