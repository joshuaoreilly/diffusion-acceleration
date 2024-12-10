import numpy as np

x = np.array([0, 0, 1, 1, 1, 0, 0])
k = np.array([1, 2, 1])

y = np.zeros(len(x))

for i in range(1, len(x)-1):
    y[i] = x[i-1] * k[0] + x[i] * k[1] + x[i+1] * k[2]

print(y)

arr = []
arr.append([0, 0, 0])
for i in range(0, len(x)-(len(k)-1)):
    arr.append([i, i+1, i+2])
arr.append([0, 0, 0])
print(arr)

arr2 = np.take(x, arr)
print(arr2)

print(arr2 @ k)