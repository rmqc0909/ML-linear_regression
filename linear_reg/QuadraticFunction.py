import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

N = 20000

X = np.linspace(0, 10, 100)
noise = np.random.normal(0, 0.1, X.shape)
Y = X * 0.5 + 3 + noise

W = 1
b = np.linspace(-5, 6, N)

h = []

for i in b:
    loss = Y - (X * W + i)
    square = np.square(loss)
    sum1 = np.sum(square)
    h.append(sum1)


print(np.min(h))
print(np.max(h))
plt.plot(b, h)

plt.savefig("F:\python\quadraticFunction.png")

plt.show()