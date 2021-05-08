import numpy as np

a = np.array(np.arange(1,4))
arr = np.array([[1,2,3],[4,5,6]])

b = a>1
print(b)
after_Arr = arr[:,b]
print(after_Arr)

print(b)
l  =sum(b)
print("--")
print(l)
c = a[b]
print(c)