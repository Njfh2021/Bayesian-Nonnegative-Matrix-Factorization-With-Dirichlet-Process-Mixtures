import numpy as np

# for x in xrange(4,10):
# 	print(x)
# def test(a,b):
# 	return a+b, a*b
# s, t = test(3,4)
# print(s)
# print(t)

a = np.array(np.arange(0, 10).reshape(2, 5))
print(a)
b = np.sum(a, axis=0) / float(2)
print("====")
print(b)
