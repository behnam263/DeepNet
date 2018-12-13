import numpy as np
import matplotlib.pyplot as plt
from nn_utils import *

def initial_parameters(layers):

    parameters ={}
    L = len(layers)

    for l in range(1,L):
        parameters['W'+ str(l)] = np.random.rand(layers[l],layers[l-1])
        parameters['b'+ str(l)] = np.random.rand(layers[l],1)

    return parameters

def single_layer_forward(A,l,parameters):

    W = parameters['W'+str(l)]
    b = parameters['b'+str(l)]
    Z = np.dot(W,A) + b

    A_next = sigmoid(Z)

    return A_next, Z

def forward_pass(X,layers,parameters):

    L = len(layers)

    caches = []
    caches.append([-1])
    A_prev = X
    for l in range(1,L):
        A_next,Z = single_layer_forward(A_prev,l,parameters)
        cache = {'A'+str(l-1):A_prev,'Z'+str(l):Z,'A'+str(l):A_next}
        caches.append(cache)
        A_prev = A_next

    return A_next, caches


def single_layer_backward(da,l,parameters,caches):

    Al = caches[l]['A'+str(l)]

    dz_l = np.multiply( Al , (1-Al))
    dA_l_1 = np.dot( parameters['W'+str(l)].T, dz_l)

    dw_l = np.dot( dz_l, caches[l]['A'+str(l-1)].T )
    db_l = np.sum(dz_l,1).reshape(-1,1)

    return dA_l_1, dw_l, db_l

def backward_pass(dA_L,layers,parameters,caches,lr):

    L = len(layers)

    dA_next = dA_L
    for l in range(L-1,0,-1):

        dA_prev, dW, db = single_layer_backward(dA_next,l,parameters,caches)
        dA_next = dA_prev

        parameters['W'+str(l)] = parameters['W'+str(l)]-lr*dW
        parameters['b' + str(l)] = parameters['b' + str(l)] - lr * db

    return parameters

#---------------------Main---------------------------
layers = [5,10,2,1]
parameters = initial_parameters(layers)


X = np.random.rand(5,100)
targ=X[1,:].reshape([100,1]).T
target = np.sin(1*2*np.pi*targ)
epoch_error = []
AL=0
target1=[]
for l in range(1,100):
    # A, Z = single_layer_forward(X,1,parameters)
    AL , caches = forward_pass(X,layers,parameters)
    #dA_L = np.random.rand(1,100)
    lr = .01
    # dA, dW, db = single_layer_backward(da,2,parameters,caches)
    AL =backward_pass(AL,layers,parameters,caches,lr)
    target1=caches[3]['Z3']
    e =  target- target1
    errors = np.array(e)
    epoch_error.append(np.sum(errors**2))
    print('Current epoch is:%s'%np.sum(errors**2))
    
#plt.plot(targ,target,'r')
#plt.plot(targ,target1)
#plt.show()

plt.figure()
plt.plot(epoch_error)