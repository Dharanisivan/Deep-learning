from keras.datasets import imdb 
from keras.models import Sequential 
from keras.layers import Embedding, LSTM, Dense 
from keras.preprocessing.sequence import pad_sequences 
# Load dataset 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) 
X_train = pad_sequences(X_train, maxlen=100) 
X_test = pad_sequences(X_test, maxlen=100) 
# Model 
model = Sequential([ 
Embedding(10000, 32, input_length=100), 
LSTM(100), 
Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2) 

  output:
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 50s 151ms/step - accuracy: 0.6617 - loss: 0.5807 - val_accuracy: 0.8452 - val_loss: 0.3591
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 88s 171ms/step - accuracy: 0.8833 - loss: 0.2813 - val_accuracy: 0.8380 - val_loss: 0.3577
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 73s 144ms/step - accuracy: 0.9172 - loss: 0.2184 - val_accuracy: 0.8382 - val_loss: 0.3901
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 82s 143ms/step - accuracy: 0.9441 - loss: 0.1678 - val_accuracy: 0.8316 - val_loss: 0.4615
Epoch 5/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 81s 141ms/step - accuracy: 0.9474 - loss: 0.1467 - val_accuracy: 0.8378 - val_loss: 0.4621
<keras.src.callbacks.history.History at 0x7c26e82431d0>
