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
# SETOSA 1.57 AND VIRGINICA -1.57
data_Alice = [ [ 5.1,3.5,1.4,0.2,1.57 ],
               [ 4.9,3.0,1.4,0.2,1.57 ],
               [ 4.7,3.2,1.3,0.2,1.57 ],
               [ 4.6,3.1,1.5,0.2,1.57 ],
               [ 5.0,3.6,1.4,0.2,1.57 ],
               [ 6.4,3.1,5.5,1.8,-1.57 ],
               [ 6.0,3.0,4.8,1.8,-1.57 ],
                [ 6.9,3.1,5.4,2.1,-1.57 ],
                [ 6.7,3.1,5.6,2.4,-1.57 ],
                [ 6.9,3.1,5.1,2.3,-1.57 ],
                [ 5.8,2.7,5.1,1.9,-1.57 ],
            ]

data_Bob = [    [ 5.4,3.9,1.7,0.4,1.57 ],
                [ 4.6,3.4,1.4,0.3,1.57 ],
                [ 5.0,3.4,1.5,0.2,1.57 ],
                [ 4.4,2.9,1.4,0.2,1.57 ],
                [ 6.9,3.1,5.4,2.1,-1.57 ],
                [ 6.7,3.1,5.6,2.4,-1.57 ],
                [ 6.9,3.1,5.1,2.3,-1.57 ],
                [ 5.4,3.7,1.5,0.2,1.57 ],
                [ 4.8,3.4,1.6,0.2,1.57 ],
                [ 4.8,3.0,1.4,0.1,1.57 ],
                [ 4.3,3.0,1.1,0.1,1.57 ],
                [ 6.4,2.7,5.3,1.9,-1.57 ],
                [ 6.8,3.0,5.5,2.1,-1.57 ],
                [ 5.7,2.5,5.0,2.0,-1.57 ],
                [ 5.8,2.8,5.1,2.4,-1.57 ],
                [ 5.8,4.0,1.2,0.2,1.57 ],
                [ 5.7,4.4,1.5,0.4,1.57 ],
                [ 5.1,3.8,1.5,0.3,1.57 ],
                [ 5.4,3.4,1.7,0.2,1.57 ],
                [ 5.1,3.7,1.5,0.4,1.57 ],
                [ 7.7,3.8,6.7,2.2,-1.57 ],
                [ 7.7,2.6,6.9,2.3,-1.57 ],
                [ 6.0,2.2,5.0,1.5,-1.57 ],
                [ 6.9,3.2,5.7,2.3,-1.57 ],
                [ 5.6,2.8,4.9,2.0,-1.57 ],
                [ 4.6,3.6,1.0,0.2,1.57 ],
                [ 5.2,4.1,1.5,0.1,1.57 ],
                [ 5.5,4.2,1.4,0.2,1.57 ],
                [ 4.9,3.1,1.5,0.1,1.57 ],
                [ 5.0,3.2,1.2,0.2,1.57 ],
                [ 5.5,3.5,1.3,0.2,1.57 ],
                [ 4.5,2.3,1.3,0.3,1.57 ],
                [ 4.8,3.0,1.4,0.3,1.57 ],
                [ 5.1,3.8,1.6,0.2,1.57 ],
                [ 4.6,3.2,1.4,0.2,1.57 ],
                [ 5.3,3.7,1.5,0.2,1.57 ],
                [ 7.1,3.0,5.9,2.0,-1.57 ],
                [ 6.3,2.9,5.6,1.8,-1.57 ],
                [ 6.5,3.0,5.8,2.2,-1.57 ],
                [ 7.6,3.0,6.6,2.1,-1.57 ],
                [ 4.9,2.5,4.5,1.7,-1.57 ],
                [ 6.4,3.2,5.3,2.3,-1.57 ],
                [ 6.5,3.0,5.5,1.8,-1.57 ],
                [ 7.7,2.8,6.7,2.0,-1.57 ],
                [ 6.3,2.7,4.9,1.8,-1.57 ], 
                [ 7.4,2.8,6.1,1.9,-1.57 ],
                [ 7.9,3.8,6.4,2.0,-1.57 ],
                [ 6.4,2.8,5.6,2.2,-1.57 ],
                [ 5.0,3.4,1.6,0.4,1.57 ],
                [ 5.2,3.5,1.5,0.2,1.57 ],
                [ 5.2,3.4,1.4,0.2,1.57 ],
                [ 4.7,3.2,1.6,0.2,1.57 ],
                [ 6.3,2.8,5.1,1.5,-1.57 ],
                [ 6.1,2.6,5.6,1.4,-1.57 ],
                [ 7.7,3.0,6.1,2.3,-1.57 ],
                [ 6.3,3.4,5.6,2.4,-1.57 ],
                [ 6.4,3.1,5.5,1.8,-1.57 ],
                [ 6.0,3.0,4.8,1.8,-1.57 ],
                [ 5.8,2.7,5.1,1.9,-1.57 ],
                [ 6.3,2.5,5.0,1.9,-1.57 ],
                [ 6.5,3.0,5.2,2.0,-1.57 ],
                [ 4.4,3.2,1.3,0.2,1.57 ],
                [ 5.0,3.5,1.6,0.6,1.57 ],
                [ 5.1,3.8,1.9,0.4,1.57 ],
 ]

data_test = [
             [ 6.2,3.4,5.4,2.3,-1.57 ],
             [ 5.9,3.0,5.1,1.8,-1.57 ],
             [ 5.4,3.9,1.3,0.4,1.57 ],
             [ 6.8,3.2,5.9,2.3,-1.57 ],
             [ 6.7,3.3,5.7,2.5,-1.57 ],
             [ 6.7,3.0,5.2,2.3,-1.57 ],
             [ 5.1,3.5,1.4,0.3,1.57 ],
             [ 4.8,3.1,1.6,0.2,1.57 ],
             [ 5.4,3.4,1.5,0.4,1.57 ],
             [ 5.7,3.8,1.7,0.3,1.57 ],
             [ 7.3,2.9,6.3,1.8,-1.57 ],
             [ 6.7,2.5,5.8,1.8,-1.57 ],
             [ 4.9,3.1,1.5,0.1,1.57 ],
             [ 4.4,3.0,1.3,0.2,1.57 ],
             [ 5.1,3.4,1.5,0.2,1.57 ],
             [ 5.0,3.5,1.3,0.3,1.57 ],
             [ 7.2,3.6,6.1,2.5,-1.57 ],
             [ 6.5,3.2,5.1,2.0,-1.57 ],
             [ 5.1,3.3,1.7,0.5,1.57 ],
             [ 4.8,3.4,1.9,0.2,1.57 ],
             [ 5.0,3.3,1.4,0.2,1.57 ],
             [ 6.3,3.3,6.0,2.5,-1.57 ],
             [ 7.2,3.2,6.0,1.8,-1.57 ],
             [ 6.2,2.8,4.8,1.8,-1.57 ],
             [ 6.1,3.0,4.9,1.8,-1.57 ],
             [ 6.4,2.8,5.6,2.1,-1.57 ],
             [ 7.2,3.0,5.8,1.6,-1.57 ],   
             [ 5.8,2.7,5.1,1.9,-1.57 ],
             [ 5.0,3.0,1.6,0.2,1.57 ],
             [ 4.9,3.1,1.5,0.1,1.57],
             
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
    def __init__(self,w1=0.3,w2=0.3,w3=0.5,w4=0.2,b=0.3,alpha=0.5):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
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
            w3 = Decimal(self.w3)
            w4 = Decimal(self.w4)
            b =  Decimal(self.b)
            l_rate = Decimal(self.learningRate)
            random_index = np.random.randint(len(data))
            point = data[random_index]

            
            z = (np.dot([w1,w2,w3,w4] , [ Decimal(point[0]),Decimal(point[1]),Decimal(point[2]),Decimal(point[3]) ])) + b

            pred = self.activation_atan(z)
            
            target = point[4]

            cost = np.square(pred - target)
            self.costs.append(cost)

            dcost_pred = Decimal(2 * (pred-target))
            dpred_dz   = Decimal(self.activation_slope(z))
            dz_dw1 = Decimal(point[0])
            dz_dw2 = Decimal(point[1])
            dz_dw3 = Decimal(point[2])
            dz_dw4 = Decimal(point[3])
            dz_db  = Decimal(1) 
     
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
            dcost_dw3 = dcost_pred * dpred_dz * dz_dw3
            dcost_dw4 = dcost_pred * dpred_dz * dz_dw4 
            dcost_db = dcost_pred * dpred_dz * dz_db

            # Back propagation and weight/bias adjustments
            t1 = Decimal(l_rate ) * Decimal(dcost_dw1)
            t2 = Decimal(l_rate ) * Decimal(dcost_dw2)
            t3 = Decimal(l_rate ) * Decimal(dcost_db)
            t4 = Decimal(l_rate ) * Decimal(dcost_dw3)
            t5 = Decimal(l_rate ) * Decimal(dcost_dw4)
            
            w1 = Decimal(w1) + Decimal(-t1)
            w2 = Decimal(w2) + Decimal(-t2)
            w3 = Decimal(w3) + Decimal(-t4)
            w4 = Decimal(w4) + Decimal(-t5)
            b  = Decimal(b) + Decimal(-t3)
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3
            self.w4 = w4
            self.b = b
        

    def encrypt_hyperparams(self):
        with localcontext() as ctx:
            ctx.prec=6
            encry_w1 = self.public_key.encrypt(Decimal(self.w1),precision=7)
            encry_w2 = self.public_key.encrypt(Decimal(self.w2),precision=7)
            encry_w3 = self.public_key.encrypt(Decimal(self.w3),precision=7)
            encry_w4 = self.public_key.encrypt(Decimal(self.w4),precision=7)
            encry_b = self.public_key.encrypt(Decimal(self.b),precision=7)
        return encry_w1,encry_w2,encry_b,encry_w3,encry_w4

    def decrypt(self,x):
        return self.private_key.decrypt(x)

    def Bobs_training(self,encrypted_z,Enc_num1,Enc_num2,Enc_num3,Enc_num4,encrypted_target):
        l_rate = Decimal(self.learningRate)
        h =  1e-6 
        for i in range(1,2):
            w1 = Decimal(self.w1)
            w2 = Decimal(self.w2)
            w3 = Decimal(self.w3)
            w4 = Decimal(self.w4)
            b = Decimal(self.b)
            
            z = self.decrypt(encrypted_z)
            num1  = self.decrypt(Enc_num1)
            num2 = self.decrypt(Enc_num2)
            num3 = self.decrypt(Enc_num3)
            num4 = self.decrypt(Enc_num4)
                         
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
            dz_dw3 = Decimal( num3 - z ) / Decimal(h)
            dz_dw4 = Decimal( num4 - z ) / Decimal(h)
            dz_db  = Decimal(1) 
     
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
            dcost_dw3 = dcost_pred * dpred_dz * dz_dw3
            dcost_dw4 = dcost_pred * dpred_dz * dz_dw4
            dcost_db = dcost_pred * dpred_dz * dz_db

            # Back propagation and weight/bias adjustments
            t1 = Decimal(l_rate ) * Decimal(dcost_dw1)
            t2 = Decimal(l_rate ) * Decimal(dcost_dw2)
            t3 = Decimal(l_rate ) * Decimal(dcost_db)
            t4 = Decimal(l_rate ) * Decimal(dcost_dw3)
            t5 = Decimal(l_rate ) * Decimal(dcost_dw4)

            w1 = Decimal(w1) + Decimal(-t1)
            w2 = Decimal(w2) + Decimal(-t2)
            b  = Decimal(b) + Decimal(-t3)
            w3 = Decimal(w3) + Decimal(-t4)
            w4 = Decimal(w4) + Decimal(-t5)
            self.w1 = w1
            self.w2 = w2
            self.b = b
            self.w3 = w3
            self.w4 = w4
            
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

    def __init__(self,Alices_public_key, encrypted_w1 , encrypted_w2,encrypted_w3,encrypted_w4, encrypted_b):
        self.public_key = Alices_public_key
        self.encrypted_w1 = encrypted_w1         
        self.encrypted_w2 = encrypted_w2
        self.encrypted_w3 = encrypted_w3         
        self.encrypted_w4 = encrypted_w4         
        self.encrypted_b = encrypted_b

    def z_calc(self,data=data_Bob):
        w1 = self.encrypted_w1
        w2 = self.encrypted_w2
        w3 = self.encrypted_w3
        w4 = self.encrypted_w4
        b =  self.encrypted_b
        h =  1e-6 
        random_index = np.random.randint(len(data))
        point = data[random_index]
    
        z = w1*point[0] + w2*point[1] + w3*point[2] + w4*point[3]+ b
        z_h_w1 = (w1+h)*point[0] + (w2)*point[1] + b + (w3)*point[2] + (w4)*point[3]+random.uniform(0.11,0.36)*h
        num1 = z_h_w1          

        z_h_w2 = (w1)*point[0] + (w2+h)*point[1] + b + (w3)*point[2] + (w4)*point[3] +random.uniform(0.11,0.36)*h
        num2 = z_h_w2          

        z_h_w3 = (w1)*point[0] + (w2)*point[1] + (w3+h)*point[2] + (w4)*point[3] + b + random.uniform(0.11,0.36)*h
        num3 = z_h_w3

        
        z_h_w4 = (w1)*point[0] + (w2)*point[1] + (w3)*point[2] + (w4+h)*point[3] + b + random.uniform(0.11,0.36)*h
        num4 = z_h_w4
        
        target = point[4]
        encrypted_target = self.public_key.encrypt(target)
        return z , num1 , num2 , num3, num4 ,encrypted_target
    
# Main function ()    
if __name__ == "__main__":

    ########################################################################
    #                        TRAINING OUR MODEL 
    # 
    ########################################################################
    # Measuring Time !
    start_time = time.time()
    
    # Alice -> 1) will train her model 2) encrypt hperprams
    #3) bob gives z_calc 4) Bobs_training
    a = Alice()
    a.training()

    for i in range(1,25000):
        W1,W2,W3,W4,B = a.encrypt_hyperparams()
        b=Bob(a.public_key,W1,W2,W3,W4,B)
        z , num1, num2 , num3,num4 ,et = b.z_calc()
        a.Bobs_training( z , num1,num2,num3,num4,et )



    ########################################################################
    #                        TESTING OUR MODEL 
    # 
    ########################################################################


    # Lets store Results into our file !     
    f = open("DataFile.txt","a")
    t =0
    fa=0
    for i in range(len(data_test)):
        point= data_test[i]
        print(point)
        z = Decimal(a.w1) * Decimal(point[0]) + Decimal(a.w2)*Decimal(point[0])+Decimal(a.w3)*Decimal(point[2]) +Decimal(a.w4)*Decimal(point[3])+ a.b
        #z = (np.dot([w1,w2] , [ Decimal(point[0]),Decimal(point[1]) ])) + b
        pred  = (a.activation_atan(z))
        print("pred : {}".format(pred))
        if (pred > 0) and (point[4] > 0) :
            f.write('TRue')
            f.write('\n')
            t=t+1

        elif (pred < 0) and (point[4] < 0) :
            f.write('TRue')
            f.write('\n')
            t=t+1

        else:
            f.write('FAlse')
            f.write('\n')
            fa=fa+1

    Accuracy = (t/(t+fa))*100        
    
    f.write('accuracy :')
    f.write(str(Accuracy))
    f.write('\n')
    f.close()
    print('Total test sets: {}'.format(t+fa))
    print('Correct predictions: {}'.format(t))
    print('Incorrect predictions: {}'.format(fa))
    print('percent accuracy : {}'.format(Accuracy))
  
    a.plot_graph()
    print("--- %s seconds ---" % (time.time() - start_time))

