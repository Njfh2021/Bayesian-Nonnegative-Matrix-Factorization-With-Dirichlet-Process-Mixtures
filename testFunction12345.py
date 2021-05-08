# a=[True]*10
# print(sum(a))
import numpy as np
a = np.array(range(1,6))
flag = a<0

print(flag)
print(sum(flag))