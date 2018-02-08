'''
 $ Author: Harsh Pathak
 $ Federated Learning 
 $ Logisitic regression (perceptron) , homomorphic encryption used 
 $ Problem in hand : Classification of a rare species flower (hypothetical scenario)
 $ Linearly seperable problem
 $ All rights to author
 $ Inspired by @Iamtrask and n1analytics and giant_neural_network

'''

# In[1] : Header Files

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random
from phe import paillier
from decimal import *
getcontext().prec = 6

# In[2] Data for each flower
data_Alice = [ [3, 1.5, 1.57],
             [2,  1 , -1.57],
             [4, 1.5, 1.57],
             [2.5, 1,   -1.57],
             [3.5, 0.5,1.57],
             [2, 0.5, -1.57],
             [5.5,1.2,1.57],
             [2.7,1,-1.57],
            ]

data_Bob = [ [3.1, 1.5, 1.57],
             [2.1,  1 , -1.57],
             [4.2, 1.5, 1.57],
             [2.8, 1,   -1.57],
             [3.5, 0.5,1.57],
             [2, 0.5, -1.57],
             [5.5,1.2,1.57],
             [1.02,1,-1.57],
            ]
data_test = [ [3.3, 1.75, 1.57],
             [2.045,  0.89 , -1.57],
             [4.4, 1.745, 1.57],
             
            ]


# In[3] Time function 
def timeCounter():
    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.3f s]' % (time.perf_counter() - time0))



'''

*******************************************************************
                             Narrative  
*******************************************************************

 Alice will train her model on her unencrypted data.
 As Alice has limited data , she will send an encrypted model to Bob who
 owns a large data warehouse ; but doesnot want anyone else to know what the
 data is . The Encrypted model will  do its learning on bob's data without
 storing it. Alice will send encrypted weights and bias to Bob to do the
 processing ; in turn Bob will return encrypted scores. Alice will then decrypt
 this score and adjust the weights and biases thereby training her model
 sucessfully 

'''

class Alice:

    costs = []
    
    # Constructor : weights , bias and learning rate
    def __init__(self,w1=0.3,w2=0.3,b=0.3,alpha=0.5):
        self.w1 = w1
        self.w2 = w2
        self.b =  b
        self.learningRate = alpha
        self.generate_keypair()


    def activation_atan(self,x):
        with localcontext() as ctx:
            ctx.prec=6
            x = Decimal(x)
            result = math.atan(x)
        return result
    
    def activation_slope(self,x):
        X = Decimal(x)
        X = X*X
        resultant = Decimal(1/(1+X))
        return resultant

    def training(self,data=data_Alice):
        for i in range(1,50):
            w1 = Decimal(self.w1)
            w2 = Decimal(self.w2)
            b =  Decimal(self.b)
            l_rate = Decimal(self.learningRate)
            random_index = np.random.randint(len(data))
            point = data[random_index]

            
            z = (np.dot([w1,w2] , [ Decimal(point[0]),Decimal(point[1]) ])) + b

            pred = self.activation_atan(z)
            
            target = point[2]

            cost = np.square(pred - target)
            self.costs.append(cost)

            dcost_pred = Decimal(2 * (pred-target))
            dpred_dz   = Decimal(self.activation_slope(z))
            dz_dw1 = Decimal(point[0])
            dz_dw2 = Decimal(point[1])
            dz_db  = Decimal(1) 
     
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
            dcost_db = dcost_pred * dpred_dz * dz_db

            # Back propagation and weight/bias adjustments
            t1 = Decimal(l_rate ) * Decimal(dcost_dw1)
            t2 = Decimal(l_rate ) * Decimal(dcost_dw2)
            t3 = Decimal(l_rate ) * Decimal(dcost_db)
    
            w1 = Decimal(w1) + Decimal(-t1)
            w2 = Decimal(w2) + Decimal(-t2)
            b  = Decimal(b) + Decimal(-t3)
            self.w1 = w1
            self.w2 = w2
            self.b = b
        

    def encrypt_hyperparams(self):
        with localcontext() as ctx:
            ctx.prec=6
            encry_w1 = self.public_key.encrypt(Decimal(self.w1),precision=7)
            encry_w2 = self.public_key.encrypt(Decimal(self.w2),precision=7)
            encry_b = self.public_key.encrypt(Decimal(self.b),precision=7)
        return encry_w1,encry_w2,encry_b

    def decrypt(self,x):
        return self.private_key.decrypt(x)

    def Bobs_training(self,encrypted_z,Enc_num1,Enc_num2,encrypted_target):
        l_rate = Decimal(self.learningRate)
        h =  1e-6 
        for i in range(1,2):
            w1 = Decimal(self.w1)
            w2 = Decimal(self.w2)
            b = Decimal(self.b)
            z = self.decrypt(encrypted_z)
            num1  = self.decrypt(Enc_num1)
            num2 = self.decrypt(Enc_num2)
                         
            target = self.decrypt(encrypted_target)

            # to mimic that i have a lot of data i will loop 
            #now lets refine the weights
            pred = self.activation_atan(z)

            cost = np.square(pred - target)
            self.costs.append(cost)

            dcost_pred = Decimal(2 * (pred-target))
            dpred_dz   = Decimal(self.activation_slope(z))
            #dz_dw1 = Decimal(31.2307692) # why this value , think a bit hint : mean
            #dz_dw2 = Decimal(1.123076923) #
            
            dz_dw1 = Decimal( num1 - z) / Decimal(h)
            dz_dw2 = Decimal( num2 - z ) / Decimal(h)
            dz_db  = Decimal(1) 
     
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
            dcost_db = dcost_pred * dpred_dz * dz_db

            # Back propagation and weight/bias adjustments
            t1 = Decimal(l_rate ) * Decimal(dcost_dw1)
            t2 = Decimal(l_rate ) * Decimal(dcost_dw2)
            t3 = Decimal(l_rate ) * Decimal(dcost_db)

            w1 = Decimal(w1) + Decimal(-t1)
            w2 = Decimal(w2) + Decimal(-t2)
            b  = Decimal(b) + Decimal(-t3)

            self.w1 = w1
            self.w2 = w2
            self.b = b
                   
    def plot_graph(self):
        plt.plot(self.costs)
        plt.title('Cost vs iterations')
        #plt.set_xlabel('iterations')
        #plt.set_ylabel('Cost ')
        plt.show()
        plt.close()
        

    def generate_keypair(self , n_length = 512):
        self.public_key,self.private_key = paillier.generate_paillier_keypair(n_length=n_length)

      
class Bob:

    def __init__(self,Alices_public_key, encrypted_w1 , encrypted_w2, encrypted_b):
        self.public_key = Alices_public_key
        self.encrypted_w1 = encrypted_w1         
        self.encrypted_w2 = encrypted_w2
        self.encrypted_b = encrypted_b

    def z_calc(self,data=data_Bob):
        w1 = self.encrypted_w1
        w2 = self.encrypted_w2
        b =  self.encrypted_b
        h =  1e-6 
        random_index = np.random.randint(len(data))
        point = data[random_index]
    
        z = w1*point[0] + w2*point[1] + b
        z_h_w1 = (w1+h)*point[0] + (w2)*point[1] + b + random.uniform(0.11,0.36)*h
        num1 = z_h_w1          

        z_h_w2 = (w1)*point[0] + (w2+h)*point[1] + b + random.uniform(0.11,0.36)*h
        num2 = z_h_w2          

        
        target = point[2]
        encrypted_target = self.public_key.encrypt(target)
        return z , num1 , num2 , encrypted_target
    
# Main function ()    
if __name__ == "__main__":
    # Alice -> 1) will train her model 2) encrypt hperprams
    #3) bob gives z_calc 4) Bobs_training
    a = Alice()
    a.training()

    for i in range(1,50000):
        W1,W2,B = a.encrypt_hyperparams()
        b=Bob(a.public_key,W1,W2,B)
        z , num1, num2 , et = b.z_calc()
        a.Bobs_training( z , num1,num2,et )

    for i in range(len(data_test)):
        point= data_test[i]
        print(point)
        z = Decimal(a.w1) * Decimal(point[0]) + Decimal(a.w2)*Decimal(point[0]) + a.b
        #z = (np.dot([w1,w2] , [ Decimal(point[0]),Decimal(point[1]) ])) + b
        pred  = (a.activation_atan(z))
        print("pred : {}".format(pred))

    a.plot_graph()
