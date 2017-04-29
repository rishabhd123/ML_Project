
# coding: utf-8

# In[4]:

from joblib import Parallel,delayed
import time
from scipy.io import loadmat 
import numpy as np
from pandas import read_csv
from multiprocessing import Process, Queue
from math import floor


# In[ ]:


# filename = 'Mug128_singlepixcam.mat'
filename = 'Ball64_singlepixcam.mat'
array = loadmat(filename)

X = np.array(array['A'])
Y = np.array(array['y'])
nr,nc = X.shape
# print X.shape
beta = 1;  # For Squared Loss
lamda = 0.216;

#Normalization-1
minimum = X.min(0);
maximum = X.max(0);

for i in np.arange(nc):
    if(maximum[i]!=minimum[i]):
        X[:,i] = (X[:,i]-minimum[i]) / (maximum[i]-minimum[i]);
    else:
        X[:,i] = np.ones(nr)

#Normalization-2
X = X - X.mean(axis=0)
nrm = np.linalg.norm(X,axis = 0);

for i in np.arange(nc):
	if nrm[i]>0:
		X[:,i] = X[:,i]/nrm[i]

Data = X
X = np.concatenate((X,-X), axis = 1)
nc = 2*nc;
# print nc,X.shape
# print nc
# sys.exit(0)

w = np.zeros(nc,dtype = np.float64).reshape(nc,1) +1
# w = np.random.rand(nc,1)

def func(index):
	# print index
	t1 = np.dot(X,w)
	t2 = t1- Y

	g = np.dot(X[:,index].T,t2) + lamda; #why nr??
	w[index] = w[index] -   0.001*g/beta; # alpha
	# print w[index]
	if w[index]<0:
		w[index] = 0;

	return w[index]
    
for i in np.arange(1):
    
    for k in np.arange(0,nc):
        w[k] = func(k)
    
    print i
    temp = nc/2
    z1 = w[temp:nc] - w[0:temp]
    z1 = z1.reshape(temp,1)

    z2 = w[0:temp]-w[temp:nc]
    z2 = z2.reshape(temp,1)
    # pred = np.dot(X , w)#.reshape(nr,1);    Change W acc to algo
    pred = np.dot(Data,z1)
    differ = (pred-Y)
    MSE = np.mean( np.square(differ) )
    print 'z1 w2-w1 MSE Error:%f' % MSE

    pred = np.dot(Data,z2)
    differ = (pred-Y)
    MSE = np.mean( np.square(differ) )
    print 'z2  w1-w2  MSE Error:%f' % MSE
    print "w"
    print np.linalg.norm(w)
    print "\n\n"



# In[2]:

temp = nc/2
z = w[temp:nc] - w[0:temp]
z = z.reshape(temp,1)
# pred = np.dot(X , w)#.reshape(nr,1);    Change W acc to algo
pred = np.dot(Data,z)
differ = (pred-Y)
MSE = np.mean( np.square(differ) );
print 'MSE Error:%f' % MSE

