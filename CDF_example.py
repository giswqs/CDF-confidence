import CDF_confidence as Cc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as ss


# Make some random discrete data
######################################################################################
# How to generate a random normal distribution of integers
# https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
x = np.arange(-10, 11)
xU, xL = x + 0.5, x - 0.5 
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
prob = prob / prob.sum() #normalize the probabilities so their sum is 1
nums = np.random.choice(x, size = 300, p = prob)
# plt.hist(nums, bins = len(x))
######################################################################################


# Make some random continuous data
rv=norm()
x=norm.rvs(size=300)
# x=nums  #uncomment this line to use random discrete data 

# Do a basic plot
plt.figure()
Cc.plot_CDF_confidence(x)

# Do fancier plot
plt.figure()
Cc.plot_CDF_confidence(x,label='Empirical CDF (90% confidence)',color='violet')

x_sorted=np.sort(x)
plt.plot(x_sorted,norm.cdf(x_sorted),':',label='True CDF')
plt.legend(loc='best')
plt.title('Pointwise Confidence Intervals')

# Do another fancier plot
plt.figure()
Cc.plot_CDF_confidence(x,label='Empirical CDF (90% confidence)',estimator_name='DKW')

x_sorted=np.sort(x)
plt.plot(x_sorted,norm.cdf(x_sorted),':',label='True CDF')
plt.legend(loc='best')
plt.title('Confidence Bands (DKW inequality)')

plt.show()
