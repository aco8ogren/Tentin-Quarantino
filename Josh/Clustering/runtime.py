import numpy as np 
a=np.array([[],[]])
# try:
b=a.mean(1)
# except:
#     print('eh')
print(b)
np.savetxt('b.txt',b)