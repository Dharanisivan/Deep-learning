from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

corpus = [
    "Deep learning is amazing",
    "Deep learning builds intelligent systems",
    "Machine learning is a subset of AI",
    "AI will change the world"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential([
    Embedding(total_words, 100, input_length=max_len-1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=1, validation_split=0.1)
output:
poch 1/50
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step - accuracy: 0.0667 - loss: 2.8332 - val_accuracy: 0.0000e+00 - val_loss: 2.8430
Epoch 2/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 736ms/step - accuracy: 0.2000 - loss: 2.8181 - val_accuracy: 0.0000e+00 - val_loss: 2.8519
Epoch 3/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.3333 - loss: 2.8029 - val_accuracy: 0.0000e+00 - val_loss: 2.8617
Epoch 4/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.2667 - loss: 2.7868 - val_accuracy: 0.0000e+00 - val_loss: 2.8731
Epoch 5/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step - accuracy: 0.2667 - loss: 2.7692 - val_accuracy: 0.0000e+00 - val_loss: 2.8868
Epoch 6/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 127ms/step - accuracy: 0.2667 - loss: 2.7493 - val_accuracy: 0.0000e+00 - val_loss: 2.9036
Epoch 7/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 146ms/step - accuracy: 0.2667 - loss: 2.7261 - val_accuracy: 0.0000e+00 - val_loss: 2.9248
Epoch 8/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 134ms/step - accuracy: 0.2000 - loss: 2.6988 - val_accuracy: 0.0000e+00 - val_loss: 2.9520
Epoch 9/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - accuracy: 0.2000 - loss: 2.6663 - val_accuracy: 0.0000e+00 - val_loss: 2.9877
Epoch 10/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.2000 - loss: 2.6273 - val_accuracy: 0.0000e+00 - val_loss: 3.0354
Epoch 11/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - accuracy: 0.2000 - loss: 2.5809 - val_accuracy: 0.0000e+00 - val_loss: 3.1007
Epoch 12/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 143ms/step - accuracy: 0.2000 - loss: 2.5266 - val_accuracy: 0.0000e+00 - val_loss: 3.1926
Epoch 13/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - accuracy: 0.2000 - loss: 2.4657 - val_accuracy: 0.0000e+00 - val_loss: 3.3249
Epoch 14/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step - accuracy: 0.2000 - loss: 2.4039 - val_accuracy: 0.0000e+00 - val_loss: 3.5184
Epoch 15/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 91ms/step - accuracy: 0.2000 - loss: 2.3541 - val_accuracy: 0.0000e+00 - val_loss: 3.7876
Epoch 16/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 91ms/step - accuracy: 0.2000 - loss: 2.3335 - val_accuracy: 0.0000e+00 - val_loss: 4.0804
Epoch 17/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step - accuracy: 0.2000 - loss: 2.3343 - val_accuracy: 0.0000e+00 - val_loss: 4.2939
Epoch 18/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 133ms/step - accuracy: 0.2000 - loss: 2.3203 - val_accuracy: 0.0000e+00 - val_loss: 4.4026
Epoch 19/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 143ms/step - accuracy: 0.2000 - loss: 2.2817 - val_accuracy: 0.0000e+00 - val_loss: 4.4374
Epoch 20/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 138ms/step - accuracy: 0.2667 - loss: 2.2325 - val_accuracy: 0.0000e+00 - val_loss: 4.4391
Epoch 21/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.2667 - loss: 2.1879 - val_accuracy: 0.0000e+00 - val_loss: 4.4411
Epoch 22/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 143ms/step - accuracy: 0.2667 - loss: 2.1542 - val_accuracy: 0.0000e+00 - val_loss: 4.4657
Epoch 23/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - accuracy: 0.2667 - loss: 2.1293 - val_accuracy: 0.0000e+00 - val_loss: 4.5254
Epoch 24/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 99ms/step - accuracy: 0.2667 - loss: 2.1076 - val_accuracy: 0.0000e+00 - val_loss: 4.6276
Epoch 25/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - accuracy: 0.3333 - loss: 2.0836 - val_accuracy: 0.0000e+00 - val_loss: 4.7782
Epoch 26/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.3333 - loss: 2.0535 - val_accuracy: 0.0000e+00 - val_loss: 4.9826
Epoch 27/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - accuracy: 0.3333 - loss: 2.0157 - val_accuracy: 0.0000e+00 - val_loss: 5.2457
Epoch 28/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - accuracy: 0.4000 - loss: 1.9712 - val_accuracy: 0.0000e+00 - val_loss: 5.5688
Epoch 29/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 136ms/step - accuracy: 0.4000 - loss: 1.9235 - val_accuracy: 0.0000e+00 - val_loss: 5.9425
Epoch 30/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step - accuracy: 0.3333 - loss: 1.8769 - val_accuracy: 0.0000e+00 - val_loss: 6.3380
Epoch 31/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 134ms/step - accuracy: 0.3333 - loss: 1.8327 - val_accuracy: 0.0000e+00 - val_loss: 6.7094
Epoch 32/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - accuracy: 0.3333 - loss: 1.7858 - val_accuracy: 0.0000e+00 - val_loss: 7.0198
Epoch 33/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.4000 - loss: 1.7320 - val_accuracy: 0.0000e+00 - val_loss: 7.2639
Epoch 34/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 147ms/step - accuracy: 0.5333 - loss: 1.6761 - val_accuracy: 0.0000e+00 - val_loss: 7.4593
Epoch 35/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - accuracy: 0.5333 - loss: 1.6237 - val_accuracy: 0.0000e+00 - val_loss: 7.6328
Epoch 36/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step - accuracy: 0.5333 - loss: 1.5684 - val_accuracy: 0.0000e+00 - val_loss: 7.8117
Epoch 37/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 136ms/step - accuracy: 0.6000 - loss: 1.5041 - val_accuracy: 0.0000e+00 - val_loss: 8.0083
Epoch 38/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step - accuracy: 0.6667 - loss: 1.4405 - val_accuracy: 0.0000e+00 - val_loss: 8.2004
Epoch 39/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 185ms/step - accuracy: 0.5333 - loss: 1.3889 - val_accuracy: 0.0000e+00 - val_loss: 8.3374
Epoch 40/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 99ms/step - accuracy: 0.6000 - loss: 1.3369 - val_accuracy: 0.0000e+00 - val_loss: 8.4151
Epoch 41/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.7333 - loss: 1.2787 - val_accuracy: 0.0000e+00 - val_loss: 8.4949
Epoch 42/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step - accuracy: 0.7333 - loss: 1.2249 - val_accuracy: 0.0000e+00 - val_loss: 8.6376
Epoch 43/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step - accuracy: 0.8000 - loss: 1.1726 - val_accuracy: 0.0000e+00 - val_loss: 8.8339
Epoch 44/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - accuracy: 0.8000 - loss: 1.1205 - val_accuracy: 0.0000e+00 - val_loss: 9.0000
Epoch 45/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.7333 - loss: 1.0763 - val_accuracy: 0.0000e+00 - val_loss: 9.0437
Epoch 46/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 97ms/step - accuracy: 0.7333 - loss: 1.0293 - val_accuracy: 0.0000e+00 - val_loss: 8.9904
Epoch 47/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 98ms/step - accuracy: 0.8000 - loss: 0.9820 - val_accuracy: 0.0000e+00 - val_loss: 8.9399
Epoch 48/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 140ms/step - accuracy: 0.8667 - loss: 0.9415 - val_accuracy: 0.0000e+00 - val_loss: 8.9448
Epoch 49/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 137ms/step - accuracy: 0.8667 - loss: 0.9014 - val_accuracy: 0.0000e+00 - val_loss: 8.9757
Epoch 50/50
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.8667 - loss: 0.8646 - val_accuracy: 0.0000e+00 - val_loss: 8.9844
<keras.src.callbacks.history.History at 0x7c26c3c259a0>

[ ]
