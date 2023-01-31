import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model, Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

data = np.loadtxt('data.txt')

input = data[:, :2]
output = data[:, -1]

C1 = input[output == 0, :]
C2 = input[output == 1, :]

plt.figure()
plt.plot(C1[:, 0], C1[:, 1], 'o')
plt.plot(C2[:, 0], C2[:, 1], '*')
plt.legend(['C1', 'C2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

output = np.reshape(output, (output.size, 1))

# Making test and training groups

input_training, input_test, output_training, output_test = train_test_split(input,
                                                                            output,
                                                                            test_size=0.2,
                                                                            shuffle=True,
                                                                            random_state=20)
plt.figure()
plt.plot(input_training[:, 0], input_training[:, 1], 'o')
plt.plot(input_test[:, 0], input_test[:, 1], '*')
plt.legend(['Training group', 'Test group'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Method 1 for creating neural network

input_layer = Input(shape=(2, ))
dense1 = Dense(10, activation='relu')(input_layer)
dense2 = Dense(5, activation='relu')(dense1)
output_layer = Dense(1, activation='sigmoid')(dense2)

model = Model(input_layer, output_layer)
model.summary()

# Method 2

model = Sequential(
    [
        Dense(10, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)
model.build((None, 2))
model.summary()

# Method 3

model = Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.build((None, 2))
model.summary()

# Setting parameters

model.compile('adam',
              loss = 'binary_crossentropy')

# Training model

max_epoch = 50

history = model.fit(x=input_training, y=output_training,
                    epochs=max_epoch,
                    batch_size=1,
                    shuffle=True,
                    verbose=0
        )
E = history.history['loss']

plt.figure()
plt.plot(E)
plt.grid()
plt.xlim([0, max_epoch])
plt.xlabel('Epoch')
plt.ylabel('$E(\omega)$')
plt.show()

output_pred = model.predict(input_test)
output_pred = np.round(output_pred)

A = np.sum(output_pred==output_test)/len(output_pred)
print(A*100)