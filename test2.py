import numpy as np
import matplotlib  
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
np.random.seed(100) 
x = np.linspace(-1, 1, 100).reshape(100,1) 
y = 3*np.power(x, 2) +2+ 0.2*np.random.rand(x.size).reshape(100,1)  
# 画图
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()