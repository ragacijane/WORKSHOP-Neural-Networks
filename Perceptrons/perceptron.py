import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

N=500

# Generating class 1
x11 = 0.5*rnd.randn(N, 1)
x21 = 0.5*rnd.randn(N, 1)
C1 = np.concatenate((x11, x21), axis=1)
D1 = np.ones((N, 1))

# Generating class 2
x12 = 0.5*rnd.randn(N, 1) + 2.5
x22 = 0.5*rnd.randn(N, 1) + 2.5
C2 = np.concatenate((x12, x22), axis=1)
D2 = np.ones((N, 1))

# Showing data
plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x12, x22, '*')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend(['C1', 'C2'])
plt.show()
# Creating input, bias will be 1

X = np.concatenate((C1, C2), axis=0)
bias = np.ones((2*N, 1))

X = np.concatenate((X, bias), axis=1)
D = np.concatenate((D1, D2), axis=0)

# Creating output function

def predict(x, w):
    act = np.sum(w * x)
    if act >= 0:
        y = 1
    else:
        y = 0

    return y

# Creating weight function

def weight_update(lr, d, y, x):
    dw = lr * (d - y) * x
    return dw

# Training parameters

W = 0.1 * rnd.randn(3) #start weight
n = 10 #Max num ofr epoch
lr = 0.01 #training constant
Emax = 0.0001 #Max err

for epoch in range(n):# Sve epohe treniranja
    error = 0
    for k in range(len(D)):# Svaki element ulaznog skupa
        Ypred = predict(W, X[k, :])
        dW = weight_update(lr, D[k], Ypred, X[k, :])
        W += dW

        error += np.abs((D[k] - Ypred))# Računanje greške klasifikacij

    error /= np.shape(X)[0]
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, lr, error))

    if error < Emax:# Ako greška padne ispod praga, zaustavlja se obučavanje
        break

w1 = W[0]
w2 = W[1]
wb = W[2]
x1 = np.linspace(-2, 6)
x2 = -w1*x1/w2 - wb/w2
plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x21, x22, '*')
plt.plot(x1,x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim([-2, 6])
plt.grid()
plt.legend(['C1', 'C2', 'h(x)=0'])
plt.show()