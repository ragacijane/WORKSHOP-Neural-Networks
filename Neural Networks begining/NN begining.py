import matplotlib.pyplot as plt
import numpy as np

#from keras import Input, Model, Sequential
#from keras.layers import Dense

from sklearn.model_selection import train_test_split

data = np.loadtxt('data.txt')

input = data[:, :2]
output = data[:, -1]

C1 = input[output==0, :]
C2 = input[output==1, :]

plt.figure()
plt.plot(C1[:, 0], C1[:, 1], 'o')
plt.plot(C2[:, 0], C2[:, 1], '*')
plt.legend(['C1','C2'])
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
plt.legend(['Training group','Test group'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()