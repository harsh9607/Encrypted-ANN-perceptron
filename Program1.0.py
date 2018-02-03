
# coding: utf-8

# In[1]:


#get_ipython().magic('matplotlib inline')


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np
import math
from decimal import * 


# In[4]:


# each point is length width and type (0,1)
data = [ [3, 1.5, 1.57],
         [2,  1 , -1.57],
         [4, 1.5, 1.57],
         [3, 1,   -1.57],
         [3.5, 0.5,1.57],
         [2, 0.5, -1.57],
         [5.5,1,1.57],
         [1,1,-1.57],
]
mystert_flower = [4.5,1]
print(len(data))


# In[5]: Encryption

from phe import paillier
public_key , private_key = paillier.generate_paillier_keypair()


# network 
#      o    flower type  
#     / \   w1 , w2, b 
#    o   o   length , width


# In[6]: Scaling


getcontext().prec = 6
W1 = 0.3
W2 = 0.3
B  = 0.3
learning_rate = 0.2

w1_ = public_key.encrypt(W1)
w2_ = public_key.encrypt(W2)
b_  = public_key.encrypt(B)
L_encrypted = public_key.encrypt(learning_rate)

w1 = w1_.ciphertext()
w2 = w2_.ciphertext()
b  = public_key.encrypt(B).ciphertext()
L_encrypted = public_key.encrypt(learning_rate).ciphertext()


def scaling(x):
    with localcontext() as ctx:
        sqrt_value = ctx.sqrt(x)
        sqrt_value2 = ctx.sqrt(sqrt_value)
        sqrt_value2 = ctx.sqrt(sqrt_value2)
        sqrt_value2 /= (10**155)
        sqrt_value2 = ctx.sqrt(sqrt_value2) 
                 
        hyperparam = sqrt_value2
        
    return hyperparam

#   Scaling before training 
w1 = scaling(w1)
w2 = scaling(w2)
b  = scaling(b)
L_encrypted = scaling(L_encrypted)
L_encrypted = Decimal(L_encrypted) / Decimal(10)



print("w1 =" ,w1)
print('w2 =' , w2)
print('b =',b)
print('alpha =', L_encrypted)

# In[7]:


def sigmoid(x):
    #Its actually tan inverse x range [-pi/2 to pi/2]
    with localcontext() as ctx :
        #print(x)
        ctx.prec = 6
        S = Decimal(x)
        pred = math.atan(S)
        #print(pred)
    return pred 

def sigmoid_p(x):
    X = Decimal(x)
    X = X*X
    resultant = Decimal(1/(1+X))
    return resultant 





# In[10]:


# In[11]:


# training loop 
costs = []
for i in range(1,100000):
    random_index = np.random.randint(len(data))
    point = data[random_index]
    
    
    
    z = (np.dot([w1,w2] , [ Decimal(point[0]),Decimal(point[1]) ])) + b
    
    pred = sigmoid(z)
    
    target = point[2]
    cost = np.square(pred - target)
    
    costs.append(cost)
    
    dcost_pred = Decimal(2 * (pred-target))
    dpred_dz   = Decimal(sigmoid_p(z))
    dz_dw1 = Decimal(point[0])
    dz_dw2 = Decimal(point[1])
    dz_db  = Decimal(1) 
     
    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db = dcost_pred * dpred_dz * dz_db
    
    
    
    '''
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
    '''
    t1 = Decimal(L_encrypted ) * Decimal(dcost_dw1)
    t2 = Decimal(L_encrypted ) * Decimal(dcost_dw2)
    t3 = Decimal(L_encrypted ) * Decimal(dcost_db)
    
    w1 = Decimal(w1) + Decimal(-t1)
    w2 = Decimal(w2) + Decimal(-t2)
    b  = Decimal(b) + Decimal(-t3)

    
plt.plot(costs)    

    

    


# In[12]:


# model prediction 
for i in range(len(data)):
    point= data[i]
    print(point)
    z = (np.dot([w1,w2] , [ Decimal(point[0]),Decimal(point[1]) ])) + b
    pred  = sigmoid(z)
    print("pred : {}".format(pred))
    
plt.title('Cost function with iterations')
plt.show()
plt.close()   
