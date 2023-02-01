import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy

data = pd.read_csv('data unbalanced classes.csv')

dataTraining, dataTest = train_test_split(data,test_size=0.2, random_state=42)

X = dataTraining.drop('d', axis=1)
Y = dataTraining['d']
Xtest = dataTest.drop(columns='d')
Ytest = dataTest.d

C0 = X.loc[Y==0, :]
C1 = X.loc[Y==1, :]

plt.figure()
plt.plot(C0.x1, C0.x2, '.')
plt.plot(C1.x1, C1.x2, '.')
plt.show()

max_epoch = 1000

plt.figure()
Y.hist()
plt.show()

C0cnt, C1cnt = np.bincount(Y)
print('Sum data: '+str(C0cnt+C1cnt))
print('Num data Class 0: '+str(C0cnt))
print('Num data Class 1: '+str(C1cnt))

def make_model(x, y):
    model = Sequential()
    model.add(Dense(10, input_dim=np.shape(x)[1], activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss=BinaryCrossentropy())
    return model

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

def calc_metrics(Xtest, Ytest, model):
    Ypred = model.predict(Xtest, verbose=0)
    Ypred = Ypred >= 0.5

    cm = confusion_matrix(Ytest, Ypred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm,
                                        display_labels=['C1', 'C2'])
    cm_display.plot()
    plt.show()

    TN = cm [0, 0]
    print('True negative: ' + str(TN))
    TP = cm[1, 1]
    print('True positive: '+str(TP))
    FN = cm[1, 0]
    print('False negative: '+str(FN))
    FP = cm[0, 1]
    print('False positive: '+str(FP))

    ACC = (TP + TN) / (TP + FP + TN + FN)
    print('Accuracy: ' + str(ACC))
    P = TP / (TP + FP)
    print('Precision: ' + str(P))
    R = TP / (TP + FN)
    print('Recal: ' + str(R))
    F1 = 2 * P * R / (P + R)
    print('F1 score: ' + str(F1))

Xtraining = X
Ytraining = Y

model=make_model(Xtraining,Ytraining)
history = model.fit(Xtraining,Ytraining,
                    epochs=max_epoch,
                    batch_size=np.shape(Xtraining)[0],
                    verbose=0)
plt.figure()
plt.plot(history.history['loss'])
plt.show()

calc_metrics(Xtest, Ytest, model)