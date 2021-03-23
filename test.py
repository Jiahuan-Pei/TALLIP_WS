# encoding=UTF-8
"""
    @author: Administrator on 2017/6/27
    @email: ppsunrise99@gmail.com
    @step:
    @function: 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)

xnew = np.arange(0, 9, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '+')
plt.show()
if __name__ == '__main__':
    pass