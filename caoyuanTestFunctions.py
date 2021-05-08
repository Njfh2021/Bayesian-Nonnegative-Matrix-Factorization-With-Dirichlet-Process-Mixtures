import numpy as np

# a= np.array([[1,2,3,4],[4,5,6,7],[8,9,10,11]])
# b= np.array([[7,8,9,10],[10,11,12,13],[14,15,16,17]])
# c = np.zeros([2,3,4])
# c[0,:,:] = a
# c[1,:,:] = b
# print(c)
# print("=======")
# print(sum(c))
M = 200
N = 200
R = 4
RU = np.random.rand(M, R)
# print(RU)
# p1= math.floor(3.5)
# print(p1)

# test = np.array([[1,2,3],[4,5,6]])
# test_1d = test.reshape(6,)
# print(test_1d)
# test_2d = test_1d.reshape([2,3])
# print(test_2d)
# ind = np.arange(5)
# print(ind)
# test[ind] = 100
# print(test)

# kmn = np.ones([2,3,4])
k = 3
m, n = 2, 4
kmn = np.reshape(np.arange(24), [k, m, n])
print(kmn)
# k_vec = np.ones(2).T
# print('-----')
# print(k_vec)

print('----------')

two_dim = np.reshape(np.arange(2 * k) + 1, [k, 2]).astype(float)
two_dim_divid = two_dim[:, 0] / two_dim[:, 1]
test = np.arange(3)
print(test[:, np.newaxis, np.newaxis])
result = kmn * test[:, np.newaxis, np.newaxis]
# print(two_dim_divid)
# result = kmn * k_vec
print(result)
print(sum(result))
