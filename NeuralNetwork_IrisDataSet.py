import numpy as np
import math
from phe import paillier
import time
import random

# Alice Training Data
data_Alice = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2],
              [5.0, 3.6, 1.4, 0.2],
              [6.4, 3.1, 5.5, 1.8],
              [6.0, 3.0, 4.8, 1.8],
              [6.9, 3.1, 5.4, 2.1],
              [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3],
              [5.8, 2.7, 5.1, 1.9]]
            )

# Alice Training Data Output
data_Alice_res = np.array([[1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0]])

data_Bob = [    [ 5.4,3.9,1.7,0.4 ],
                [ 4.6,3.4,1.4,0.3 ],
                [ 5.0,3.4,1.5,0.2 ],
                [ 4.4,2.9,1.4,0.2 ],
                [ 6.9,3.1,5.4,2.1 ],
                [ 6.7,3.1,5.6,2.4 ],
                [ 6.9,3.1,5.1,2.3 ],
                [ 5.4,3.7,1.5,0.2 ],
                [ 4.8,3.4,1.6,0.2 ],
                [ 4.8,3.0,1.4,0.1 ],
                [ 4.3,3.0,1.1,0.1 ],
                [ 6.4,2.7,5.3,1.9 ],
                [ 6.8,3.0,5.5,2.1 ],
                [ 5.7,2.5,5.0,2.0 ],
                [ 5.8,2.8,5.1,2.4 ],
                [ 5.8,4.0,1.2,0.2 ],
                [ 5.7,4.4,1.5,0.4 ],
                [ 5.1,3.8,1.5,0.3 ],
                [ 5.4,3.4,1.7,0.2 ],
                [ 5.1,3.7,1.5,0.4 ],
                [ 7.7,3.8,6.7,2.2 ],
                [ 7.7,2.6,6.9,2.3 ],
                [ 6.0,2.2,5.0,1.5 ],
                [ 6.9,3.2,5.7,2.3 ],
                [ 5.6,2.8,4.9,2.0 ],
                [ 4.6,3.6,1.0,0.2 ],
                [ 5.2,4.1,1.5,0.1 ],
                [ 5.5,4.2,1.4,0.2 ],
                [ 4.9,3.1,1.5,0.1 ],
                [ 5.0,3.2,1.2,0.2 ],
                [ 5.5,3.5,1.3,0.2 ],
                [ 4.5,2.3,1.3,0.3 ],
                [ 4.8,3.0,1.4,0.3 ],
                [ 5.1,3.8,1.6,0.2 ],
                [ 4.6,3.2,1.4,0.2 ],
                [ 5.3,3.7,1.5,0.2 ],
                [ 7.1,3.0,5.9,2.0 ],
                [ 6.3,2.9,5.6,1.8 ],
                [ 6.5,3.0,5.8,2.2 ],
                [ 7.6,3.0,6.6,2.1 ],
                [ 4.9,2.5,4.5,1.7 ],
                [ 6.4,3.2,5.3,2.3 ],
                [ 6.5,3.0,5.5,1.8 ],
                [ 7.7,2.8,6.7,2.0 ],
                [ 6.3,2.7,4.9,1.8 ], 
                [ 7.4,2.8,6.1,1.9 ],
                [ 7.9,3.8,6.4,2.0 ],
                [ 6.4,2.8,5.6,2.2 ],
                [ 5.0,3.4,1.6,0.4 ],
                [ 5.2,3.5,1.5,0.2 ],
                [ 5.2,3.4,1.4,0.2 ],
                [ 4.7,3.2,1.6,0.2 ],
                [ 6.3,2.8,5.1,1.5 ],
                [ 6.1,2.6,5.6,1.4 ],
                [ 7.7,3.0,6.1,2.3 ],
                [ 6.3,3.4,5.6,2.4 ],
                [ 6.4,3.1,5.5,1.8 ],
                [ 6.0,3.0,4.8,1.8 ],
                [ 5.8,2.7,5.1,1.9 ],
                [ 6.3,2.5,5.0,1.9 ],
                [ 6.5,3.0,5.2,2.0 ],
                [ 4.4,3.2,1.3,0.2 ],
                [ 5.0,3.5,1.6,0.6 ],
                [ 5.1,3.8,1.9,0.4 ],
 ]

data_Bob_res = np.array([[1], [1],[1], [1], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [0], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1]])

# Test Data
data_test = np.array([
                    [6.2, 3.4, 5.4, 2.3],
                    [5.9, 3.0, 5.1, 1.8],
                    [5.4, 3.9, 1.3, 0.4],
                    [6.8, 3.2, 5.9, 2.3],
                    [6.7, 3.3, 5.7, 2.5],
                    [6.7, 3.0, 5.2, 2.3],
                    [5.1, 3.5, 1.4, 0.3],
                    [4.8, 3.1, 1.6, 0.2],
                    [5.4, 3.4, 1.5, 0.4],
                    [5.7, 3.8, 1.7, 0.3],
                    [7.3, 2.9, 6.3, 1.8],
                    [6.7, 2.5, 5.8, 1.8],
                    [4.9, 3.1, 1.5, 0.1],
                    [4.4, 3.0, 1.3, 0.2],
                    [5.1, 3.4, 1.5, 0.2],
                    [5.0, 3.5, 1.3, 0.3],
                    [7.2, 3.6, 6.1, 2.5],
                    [6.5, 3.2, 5.1, 2.0],
                    [5.1, 3.3, 1.7, 0.5],
                    [4.8, 3.4, 1.9, 0.2],
                    [5.0, 3.3, 1.4, 0.2],
                    [6.3, 3.3, 6.0, 2.5],
                    [7.2, 3.2, 6.0, 1.8],
                    [6.2, 2.8, 4.8, 1.8],
                    [6.1, 3.0, 4.9, 1.8],
                    [6.4, 2.8, 5.6, 2.1],
                    [7.2, 3.0, 5.8, 1.6],
                    [5.8, 2.7, 5.1, 1.9],
                    [5.0, 3.0, 1.6, 0.2],
                    [4.9, 3.1, 1.5, 0.1]]
                )

# Test Data Output
data_test_res = np.array([[0],[0],[1],[0],[1],[0],[1],[1],[1],[1],[0],[0],[1],[1],[1],[1],[0],[0],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[1],[1]])

# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Atan Function
def activation_atan(x):
    result = np.arctan(x)
    return result
    
# Derivative of Atan Function
def activation_slope(x):
    X = x*x
    resultant = (1/(1+X))
    return resultant

# Class initialization
class Alice:

    def __init__(self):
        self.epoch = 50000  # Setting training iterations
        self.lr = 0.001  # Setting learning rate
        self.inputlayer_neurons = data_Alice.shape[1]  # number of features in data set
        self.hiddenlayer_neurons = 5 # number of hidden layers neurons
        self.output_neurons = 1  # number of neurons at output layer
        self.wh = np.random.uniform(size=(self.inputlayer_neurons, self.hiddenlayer_neurons))
        self.bh = np.random.uniform(size=(1, self.hiddenlayer_neurons))
        self.wout = np.random.uniform(size=(self.hiddenlayer_neurons, self.output_neurons))
        self.bout = np.random.uniform(size=(1, self.output_neurons))
        self.generate_keypair()
        # print("Currently in Alice Training")
        # print("Shape of wh = ", self.wh.shape)
        # print("Shape of bh = ", self.bh.shape)
        # print("Shape of wout = ", self.wout.shape)
        # print("Shape of bout= ", self.bout.shape)


    def training(self,data=data_Alice):
        for i in range(self.epoch):
    
            # Forward Propogation
            hidden_layer_input1 = np.dot(data, self.wh)
            hidden_layer_input = hidden_layer_input1 + self.bh
            hiddenlayer_activations = sigmoid(hidden_layer_input)
            output_layer_input1 = np.dot(hiddenlayer_activations, self.wout)
            output_layer_input = output_layer_input1 + self.bout
            output = sigmoid(output_layer_input)

            # Backpropagation
            E = data_Alice_res-output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(self.wout.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            self.wout += hiddenlayer_activations.T.dot(d_output) * self.lr
            self.bout += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.wh += data_Alice.T.dot(d_hiddenlayer) * self.lr
            self.bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * self.lr
        

    def decrypt(self,x):
        return self.private_key.decrypt(x)

    def generate_keypair(self , n_length = 512):
        self.public_key,self.private_key = paillier.generate_paillier_keypair(n_length=n_length)

    def encrypt_hyperparams(self):
        encrypt = self.public_key.encrypt(1,precision=10)
        encry_wh = np.dot(encrypt,self.wh)
        encry_bh = np.dot(encrypt,self.bh)
        encry_wout = np.dot(encrypt,self.wout)
        encry_bout = np.dot(encrypt,self.bout)
        return encry_wh,encry_bh,encry_wout,encry_bout

    def iter_function(arr,i):
            return arr[0][i]
                
    def Bobs_training(self,encrypted_z,Enc_num1,Enc_num2,Enc_num3,Enc_num4,encrypted_target):
        l_rate = (self.lr)
        h =  1e-6 
        for i in range(1,2):
            wh = self.wh
            bh = self.bh
            wout = self.wout
            bout = self.bout
            # print("Currently in Bob Training")
            # print("Shape of wh = ", wh.shape)
            # print("Shape of bh = ", bh.shape)
            # print("Shape of wout = ", wout.shape)
            # print("Shape of bout = ", bout.shape)
            
            z = np.ndarray(shape=(1,self.hiddenlayer_neurons))
            for i in range(encrypted_z.shape[0]):
                for j in range(encrypted_z.shape[1]):
                    z[i][j]=(self.decrypt(encrypted_z[i][j]))
            

            num1 = np.ndarray(shape=(1,self.hiddenlayer_neurons))
            for i in range(Enc_num1.shape[0]):
                for j in range(Enc_num1.shape[1]):
                    num1[i][j]=(self.decrypt(Enc_num1[i][j]))

            num2 = np.ndarray(shape=(1,self.hiddenlayer_neurons))
            for i in range(Enc_num2.shape[0]):
                for j in range(Enc_num2.shape[1]):
                    num2[i][j]=(self.decrypt(Enc_num2[i][j]))

            num3 = np.ndarray(shape=(1,self.hiddenlayer_neurons))
            for i in range(Enc_num3.shape[0]):
                for j in range(Enc_num3.shape[1]):
                    num3[i][j]=(self.decrypt(Enc_num3[i][j]))

            num4 = np.ndarray(shape=(1,self.hiddenlayer_neurons))
            for i in range(Enc_num4.shape[0]):
                for j in range(Enc_num4.shape[1]):
                    num4[i][j]=(self.decrypt(Enc_num4[i][j]))

            encrypted_target = encrypted_target[0]
            target = self.decrypt(encrypted_target)

            hiddenlayer_activations = sigmoid(z)
            output_layer_input1 = np.dot(hiddenlayer_activations, self.wout)
            output_layer_input = output_layer_input1 + self.bout
            output = sigmoid(output_layer_input)

            dz_dwh=np.zeros(shape=(1,self.inputlayer_neurons))
            # print(dz_dwh[0])
            for j in range(z.shape[1]): dz_dwh[0][0] +=(num1[0][j] - z[0][j]) / (h)
            for j in range(z.shape[1]): dz_dwh[0][1] +=(num2[0][j] - z[0][j]) / (h)
            for j in range(z.shape[1]): dz_dwh[0][2] +=(num3[0][j] - z[0][j]) / (h)
            for j in range(z.shape[1]): dz_dwh[0][3] +=(num4[0][j] - z[0][j]) / (h)

            E = target-output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(self.wout.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            self.wout += hiddenlayer_activations.T.dot(d_output) * self.lr
            self.bout += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.wh += dz_dwh.T.dot(d_hiddenlayer) * self.lr
            self.bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * self.lr

            self.wh = wh
            self.bh = bh
            self.wout = wout
            self.bout = bout

class Bob:
    
    def __init__(self,Alices_public_key, encrypted_wh , encrypted_bh,encrypted_wout,encrypted_bout):
        self.public_key = Alices_public_key
        self.encrypted_wh = encrypted_wh
        self.encrypted_bh = encrypted_bh
        self.encrypted_wout = encrypted_wout         
        self.encrypted_bout = encrypted_bout        

    def z_calc(self,data=data_Bob):
        wh = self.encrypted_wh
        bh = self.encrypted_bh
        wout = self.encrypted_wout
        bout = self.encrypted_bout
        h =  1e-6 
        random_index = np.random.randint(len(data))
        point = data[random_index]
    
        hidden_layer_input1 = np.dot(point, wh)
        hidden_layer_input=hidden_layer_input1 + bh
        z = hidden_layer_input

        hidden_layer_input1 = point[0]*(wh[0]+h)+point[1]*wh[1]+point[2]*wh[2]+point[3]*wh[3]
        hidden_layer_input=hidden_layer_input1 + bh + random.uniform(0.11,0.36)*h
        num1 = hidden_layer_input

        hidden_layer_input1 = point[0]*wh[0]+point[1]*(wh[1]+h)+point[2]*wh[2]+point[3]*wh[3]
        hidden_layer_input=hidden_layer_input1 + bh + random.uniform(0.11,0.36)*h
        num2 = hidden_layer_input  

        hidden_layer_input1 = point[0]*wh[0]+point[1]*wh[1]+point[2]*(wh[2]+h)+point[3]*wh[3]
        hidden_layer_input=hidden_layer_input1 + bh + random.uniform(0.11,0.36)*h
        num3 = hidden_layer_input  

        hidden_layer_input1 = point[0]*wh[0]+point[1]*wh[1]+point[2]*wh[2]+point[3]*(wh[3]+h)
        hidden_layer_input=hidden_layer_input1 + bh + random.uniform(0.11,0.36)*h
        num4 = hidden_layer_input                
        
        target = data_Bob_res[random_index]
        encryp = self.public_key.encrypt(1,precision=7)
        encrypted_target = np.dot(encryp,target)
        return z , num1 , num2 , num3, num4 ,encrypted_target

# Main function ()
if __name__ == "__main__":
        
    start_time = time.time()
    # Alice -> 1) will train her model 2) encrypt hperprams
    #3) bob gives z_calc 4) Bobs_training
    a = Alice()
    a.training()
    print("Alice Training Completed")

    for i in range(1,5000):
        WH,BH,WOUT,BOUT = a.encrypt_hyperparams()
        b=Bob(a.public_key,WH,BH,WOUT,BOUT)
        z , num1, num2 , num3,num4 ,et = b.z_calc()
        a.Bobs_training( z , num1,num2,num3,num4,et )
    print("Bob Training Completed")
    print("--- %s seconds ---" % (time.time() - start_time))


########################################################################
#                        TESTING OUR MODEL 
########################################################################

start_time = time.time()
t = 0
for i in range(len(data_test)):
    point = data_test[i]
    # Forward Propogation
    hidden_layer_input1 = np.dot(point, a.wh)
    hidden_layer_input=hidden_layer_input1 + a.bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,a.wout)
    output_layer_input= output_layer_input1 + a.bout
    output = sigmoid(output_layer_input)
    output = output[0][0]
    print("pred : {}".format(output))
    if (round(output) == data_test_res[i]) :
        print ("True")
        t=t+1
    else:
        print ("False")
        
Accuracy = float(t/30.0)*100   
print (Accuracy)
print("--- %s seconds ---" % (time.time() - start_time))