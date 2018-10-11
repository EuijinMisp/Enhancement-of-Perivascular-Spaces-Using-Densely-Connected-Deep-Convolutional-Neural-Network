# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:23:52 2018

@author: Owner
#modification
2d conv-> 3dconv
no deconv
output -> tanh*7 :  max intensity =~ 6.xx    
batch_normalize
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as slim

def Concatenation(layers,ax):
    return tf.concat(layers, axis=ax)
   
def SkipConnect(conv,ax):
    skipconv = list()
    for i in conv:
        x = Concatenation(i,ax)
        skipconv.append(x)
    return skipconv



class DDCNN(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        

    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    if i<=2:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                    else:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate*(i-1), 8], stddev=np.sqrt(2.0/(fs*fs*fs)/self.growth_rate*(i-1))), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        convin_last=list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)
            convin_last.append(conv_in[-1])
            nextlayer = Concatenation(convin_last,4)     
            conv.append(conv_in)
            
        return conv[-1]

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        x = SkipConnect(x,4)
        x = Concatenation(x,4)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim, 8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight

        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H 


        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, allfeature, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred

class DDCNN_SC(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        

    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    if i<=2:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                    else:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate*(i-1), 8], stddev=np.sqrt(2.0/(fs*fs*fs)/self.growth_rate*(i-1))), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        convin_last=list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)
            convin_last.append(conv_in[-1])
            nextlayer = Concatenation(convin_last,4)     
            conv.append(conv_in)
            
        return conv
    """
    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 
    """
    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block
        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        #x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim, 8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8

        #self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        #self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, allfeature, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred

class DDCNN_SC_B(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        

    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    if i<=2:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                    else:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate*(i-1), 8], stddev=np.sqrt(2.0/(fs*fs*fs)/self.growth_rate*(i-1))), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        convin_last=list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)
            convin_last.append(conv_in[-1])
            nextlayer = Concatenation(convin_last,4)     
            conv.append(conv_in)
            
        return conv
    
    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 
    
    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block
        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim, 8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 16, self.c_dim], stddev = np.sqrt(2.0/27/16)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred
class RDDCNN_SC_B(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        

    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    if i<=2:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                    else:
                        weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate*(i-1), 8], stddev=np.sqrt(2.0/(fs*fs*fs)/self.growth_rate*(i-1))), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        convin_last=list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)
            convin_last.append(conv_in[-1])
            nextlayer = Concatenation(convin_last,4)     
            conv.append(conv_in)
            
        return conv
    
    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 
    
    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block
        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.reconv_layer(x)
        x = x +self.images

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim, 8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 16, self.c_dim], stddev = np.sqrt(2.0/27/16)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred

class RDensenet(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer = conv_in[-1]
            
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.reconv_layer(x)
        x = self.images+x

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * (self.des_block_H) * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 16, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred 

class RDensenet_Symskip(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        sym_skip=list()
        conv = list()
        skip_level=3
        for i in range(1, outlayer+1):            
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer = conv_in[-1]
            if i<int(outlayer/2) :
                sym_skip.append(nextlayer)
            elif (i>int(outlayer/2)) and (i<outlayer):
                nextlayer = nextlayer+sym_skip[i-skip_level]
                skip_level+=2

            
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.low_conv + x
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * (self.des_block_H) * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 8], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([8], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 8, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred 

class RDensenet_Symskip2(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        sym_skip=list()
        conv = list()
        skip_level=3
        for i in range(1, outlayer+1):            
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer = conv_in[-1]
            if i<int(outlayer/2) :
                sym_skip.append(nextlayer)
            elif (i>int(outlayer/2)) and (i<outlayer):
                nextlayer = nextlayer+sym_skip[i-skip_level]
                skip_level+=2

            
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.reconv_layer(x)
        x = x+self.images

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * (self.des_block_H) * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 8], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([8], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 8, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred 

class DCNN6_SC_B(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer =conv_in[-1]
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, 16, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred
class DCNN1(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    def BN(self,input):

        mean, var = tf.nn.moments(input, axes = [0,1,2,3])
        output = tf.nn.batch_normalization(input, mean = mean, variance = var, offset = 0, scale = 1, variance_epsilon = 0.01)

        return output


    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
            
        weightsH.update({'w_H_1': tf.Variable(tf.random_normal([fs, fs, fs, 1, 200], stddev=np.sqrt(2.0/(fs*fs*fs)/1)), name='w_H_1')}) 
        biasesH.update({'b_H_1': tf.Variable(tf.zeros([200], name='b_H_1' ))})

        weightsH.update({'w_H_2': tf.Variable(tf.random_normal([fs, fs, fs, 200, 48], stddev=np.sqrt(2.0/(fs*fs*fs)/(200))), name='w_H_2')}) 
        biasesH.update({'b_H_2': tf.Variable(tf.zeros([48], name='b_H_2' ))})    
        
        weightsH.update({'w_H_3': tf.Variable(tf.random_normal([fs, fs, fs, 248, self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/248)), name='w_H_3')}) 
        biasesH.update({'b_H_3': tf.Variable(tf.zeros([self.growth_rate], name='b_H_3' ))})

        weightsH.update({'w_H_4': tf.Variable(tf.random_normal([fs, fs, fs, 272, self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/272)), name='w_H_4')}) 
        biasesH.update({'b_H_4': tf.Variable(tf.zeros([self.growth_rate], name='b_H_4' ))})

        weightsH.update({'w_H_5': tf.Variable(tf.random_normal([fs, fs, fs, 296, self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/296)), name='w_H_5')}) 
        biasesH.update({'b_H_5': tf.Variable(tf.zeros([self.growth_rate], name='b_H_5' ))})

        weightsH.update({'w_H_6': tf.Variable(tf.random_normal([fs, fs, fs, 320, 1], stddev=np.sqrt(2.0/(fs*fs*fs)/320)), name='w_H_6')})        
        biasesH.update({'b_H_6': tf.Variable(tf.zeros([1], name='b_H_6' ))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self):
        conv_in = list()               

        # The first conv need connect with low level layer
        x = tf.nn.conv3d(self.images, self.weight_block['w_H_1'], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_1']
        conv_in.append(x)

        for i in range(2,len(self.weight_block)):
            x = Concatenation(conv_in,4)
            x = tf.nn.conv3d(x, self.weight_block['w_H_%d'%i], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d'%i]               
            x = self.BN(x)
            x = tf.nn.relu(x)
            conv_in.append(x)
        
        x = Concatenation(conv_in,4)
        x = tf.nn.conv3d(x, self.weight_block['w_H_6'], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_6']   

        return x
    

        
    def model(self):
        x = self.desBlock()
        return x

    def build_model(self):
           
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH()       
        self.pred = self.model()
        
        
        return self.pred
class DCNN(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        #conv = list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer =conv_in[-1]
            #conv.append(conv_in)
        return conv_in
    
    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 
    
    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x = Concatenation(x,4)
        #x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H

        #self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        #self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, allfeature, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred


class DCNN_SC(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        skip_low_layer=list()
        for i in range(1, outlayer+1):
            conv_in = list()               
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv3d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,4)
                    x = tf.nn.conv3d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]               
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer =conv_in[-1]
            conv.append(conv_in)
        return conv
    
    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 
    
    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,4)
        x.append(self.low_conv)
        x = Concatenation(x,4)
        #x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, 3, self.c_dim,8], stddev= np.sqrt(2.0/27/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv3d(self.images, self.low_weight, strides=[1,1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        
            #16 -> 128 -> 1024 
        
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8
        
        #self.bot_weight = tf.Variable(tf.random_normal([1, 1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        #self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))
        
        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, 3, allfeature, self.c_dim], stddev = np.sqrt(2.0/27/allfeature)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred


class SRCNN(object):

     def __init__(self,images):
    
        self.images = images 
        self.build_model()
    
     def build_model(self):
       
        self.weights = {
          'w1': tf.Variable(tf.random_normal([9, 9, 9, 1, 64], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([1, 1, 1, 64, 32], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([5, 5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
          'b1': tf.Variable(tf.zeros([64]), name='b1'),
          'b2': tf.Variable(tf.zeros([32]), name='b2'),
          'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
    
        self.pred = self.model()
        
        return self.pred

     def model(self):
        
        conv1 = tf.nn.relu(tf.nn.conv3d(self.images, self.weights['w1'], strides=[1,1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv3d(conv1, self.weights['w2'], strides=[1,1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv3d(conv2, self.weights['w3'], strides=[1,1,1,1,1], padding='SAME') + self.biases['b3']

        return conv3


class VDSR(object):
    def __init__(self,input_tensor,num_layers,n_channel=32):
        print("Building VDSR...")

        self.input_tensor = input_tensor
        self.num_layers = num_layers
        self.n_channel = n_channel
        self.weightsH = {}
        self.biasesH = {}
        self.build_model()


    def build_model(self):
        
        self.weightsH.update({'conv_01_w': tf.Variable(tf.random_normal([3, 3, 3, 1, self.n_channel], stddev=np.sqrt(2.0/27)), name='conv_01_w')})
        self.biasesH.update({'conv_01_b': tf.Variable(tf.zeros([self.n_channel], name='conv_01_b'))})
     

        for i in range(2,self.num_layers):
            self.weightsH.update({'conv_%02d_w'%i: tf.Variable(tf.random_normal([3,3,3,self.n_channel,self.n_channel], stddev=np.sqrt(2.0/27/self.n_channel)), name='conv_%02d_w'%i)})
            self.biasesH.update({'conv_%02d_b'%i: tf.Variable(tf.zeros([self.n_channel], name='conv_%02d_b'%i))})
          
        self.weightsH.update({'conv_'+str(self.num_layers)+'_w': tf.Variable(tf.random_normal([3, 3, 3, self.n_channel, 1], stddev=np.sqrt(2.0/27)), name='conv_'+str(self.num_layers)+'_w')})
        self.biasesH.update({'conv_'+str(self.num_layers)+'_b': tf.Variable(tf.zeros([1], name='conv_'+str(self.num_layers)+'_b'))})

        predict = self.model()

        return predict

    def model(self):

        tensor = tf.nn.relu(tf.nn.conv3d(self.input_tensor, self.weightsH['conv_01_w'], strides=[1,1,1,1,1], padding='SAME')+ self.biasesH['conv_01_b'])

        for i in range(2,self.num_layers):
            tensor = tf.nn.relu(tf.nn.conv3d(tensor, self.weightsH['conv_%02d_w'%i], strides=[1,1,1,1,1], padding='SAME')+ self.biasesH['conv_%02d_b'%i])
          
        tensor = tf.nn.conv3d(tensor, self.weightsH['conv_'+str(self.num_layers)+'_w'], strides=[1,1,1,1,1], padding='SAME')+ self.biasesH['conv_'+str(self.num_layers)+'_b']

        tensor = tf.add(tensor, self.input_tensor)

        return tensor

class EDSR(object):

    def __init__(self,input_x,num_blocks,num_layers,feature_size=32):
        
        print("Building EDSR...")

        self.input = input_x
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.n_channel = feature_size
        self.build_model()

    def build_model(self):

        self.weightsH = {}
        self.biasesH = {}
        self.weightsH.update({'low_w': tf.Variable(tf.random_normal([3, 3, 3, 1, self.n_channel], stddev=np.sqrt(2.0/27)), name='low_w')})
        self.biasesH.update({'low_b': tf.Variable(tf.zeros([self.n_channel], name='low_b'))})
        
        for i in range(1,self.num_blocks+1):
            for j in range(1,self.num_layers+1):           
                self.weightsH.update({'w_%d_%d'%(i,j): tf.Variable(tf.random_normal([3, 3, 3, self.n_channel, self.n_channel], stddev=np.sqrt(2.0/27/self.n_channel)), name='w_%d_%d'%(i,j))})
                self.biasesH.update({'b_%d_%d'%(i,j): tf.Variable(tf.zeros([self.n_channel], name='b_%d_%d'%(i,j)))})
            

        self.weightsH.update({'w_reconv1': tf.Variable(tf.random_normal([3, 3, 3, self.n_channel, self.n_channel], stddev=np.sqrt(2.0/27/self.n_channel)), name='w_reconv1')})
        self.biasesH.update({'b_reconv1': tf.Variable(tf.zeros([self.n_channel], name='b_reconv1'))})

        self.weightsH.update({'w_reconv2': tf.Variable(tf.random_normal([3, 3, 3, self.n_channel, 1], stddev=np.sqrt(2.0/27/self.n_channel)), name='w_reconv2')})
        self.biasesH.update({'b_reconv2': tf.Variable(tf.zeros([self.n_channel], name='b_reconv2'))})

        self.prediction = self.model()

        return self.prediction

        
    def resBlock(self,x,scale):

        for i in range(1,self.num_blocks+1):
            for j in range(1,self.num_layers+1):
            
                tmp = tf.nn.conv3d(x, self.weightsH['w_%d_%d'%(i,j)], strides=[1,1,1,1,1], padding='SAME')+self.biasesH['b_%d_%d'%(i,j)]
                
                if j < self.num_layers:
                    tmp = tf.nn.relu(tmp)

            tmp *= scale
            x = x + tmp
        
        return  x
    
    def model(self): 

        scaling_factor = 0.1

        x = tf.nn.conv3d(self.input, self.weightsH['low_w'], strides=[1,1,1,1,1], padding='SAME') + self.biasesH['low_b']

        conv_1 = x        
       
        x = self.resBlock(x,scale=scaling_factor)
        x = tf.nn.conv3d(x,self.weightsH['w_reconv1'], strides=[1,1,1,1,1], padding='SAME')+self.biasesH['b_reconv1']
        x += conv_1
        x = tf.nn.conv3d(x,self.weightsH['w_reconv2'], strides=[1,1,1,1,1], padding='SAME')+self.biasesH['b_reconv2']

        self.out = x

        return self.out


class DCNN_2d(object):

    def __init__(self,
                 input_x,
                 c_dim,
                 des_block_H,
                 des_block_ALL,
                 growth_rate,
                 filter_size
                 ):


        self.c_dim = c_dim
        self.des_block_H = des_block_H
        self.des_block_ALL = des_block_ALL
        self.growth_rate = growth_rate
        self.images = input_x
        self.filter_size = filter_size
        self.build_model()
        
    
    # Create DenseNet init weight and biases, init_block mean the growrate
    def DesWBH(self, desBlock_layer, outlayer):
        weightsH = {}
        biasesH = {}
        fs = self.filter_size
        for i in range(1, outlayer+1):
            for j in range(1, desBlock_layer+1):
                if j is 1:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, 8, 8], stddev=np.sqrt(2.0/(fs*fs)/8)), name='w_H_%d_%d' % (i, j))}) 
                else:
                    weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, self.growth_rate * (j-1), self.growth_rate], stddev=np.sqrt(2.0/(fs*fs)/(self.growth_rate * (j-1)))), name='w_H_%d_%d' % (i, j))}) 
                biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([self.growth_rate], name='b_H_%d_%d' % (i, j)))})
        return weightsH, biasesH
    

    # Create one Dense Block Convolution Layer 
    def desBlock(self, desBlock_layer, outlayer):
        nextlayer = self.low_conv
        conv = list()
        for i in range(1, outlayer+1):
            conv_in = list()
            for j in range(1, desBlock_layer+1):
                # The first conv need connect with low level layer
                if j is 1:
                    x = tf.nn.conv2d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self.biases_block['b_H_%d_%d' % (i, j)]
                    x = tf.nn.relu(x)
                    conv_in.append(x)
                else:
                    x = Concatenation(conv_in,3)
                    x = tf.nn.conv2d(x, self.weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1], padding='SAME')+ self.biases_block['b_H_%d_%d' % (i, j)]             
                    x = tf.nn.relu(x)
                    conv_in.append(x)

            nextlayer = conv_in[-1]
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.bot_weight, strides=[1,1,1,1], padding='SAME') + self.bot_biases                                  
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv2d(input_layer, self.reconv_weight, strides=[1,1,1,1], padding='SAME') + self.reconv_biases
        return x 

        
        
    def model(self):
        x = self.desBlock(self.des_block_H, self.des_block_ALL)
        # NOTE: Cocate all dense block

        x = SkipConnect(x,3)
        x.append(self.low_conv)
        x = Concatenation(x,3)
        #x = self.bot_layer(x)
        x = self.reconv_layer(x)

        return x


    def build_model(self):
        
              
        # Low Level Layer
        self.low_weight = tf.Variable(tf.random_normal([3, 3, self.c_dim, 8], stddev= np.sqrt(2.0/9/self.c_dim)), name='w_low')
        self.low_biases = tf.Variable(tf.zeros([8], name='b_low'))
        self.low_conv = tf.nn.relu(tf.nn.conv2d(self.images, self.low_weight, strides=[1,1,1,1], padding='SAME') + self.low_biases)
        
        # NOTE: Init each block weight
        """
            16 -> 128 -> 1024 
        """
        # DenseNet blocks 
        self.weight_block, self.biases_block = self.DesWBH(self.des_block_H,self.des_block_ALL)

        # Bottleneck layer
        allfeature = self.growth_rate * self.des_block_H * self.des_block_ALL + 8

        self.bot_weight = tf.Variable(tf.random_normal([1, 1, allfeature, 16], stddev = np.sqrt(2.0/1/allfeature)), name='w_bot')
        self.bot_biases = tf.Variable(tf.zeros([16], name='b_bot'))

        # Reconstruction layer

        self.reconv_weight = tf.Variable(tf.random_normal([3, 3, allfeature, self.c_dim], stddev = np.sqrt(2.0/9/16)), name ='w_reconv')
        self.reconv_biases = tf.Variable(tf.zeros([self.c_dim], name='b_reconv'))
        
        self.pred = self.model()
        
        
        return self.pred
class SRCNN2d(object):

     def __init__(self,images):
    
        self.images = images
        self.build_model() 

    
     def build_model(self):
       
        self.weights = {
          'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
          'b1': tf.Variable(tf.zeros([64]), name='b1'),
          'b2': tf.Variable(tf.zeros([32]), name='b2'),
          'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
    
        self.pred = self.model()
        
        return self.pred

     def model(self):
        
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']

        return conv3
 

class U_NET(object):  
    
    def __init__(self, x, batch_):
        
        self.x = x
        self.batch_ = batch_
        
    def model(self):
        
        m_conv1_W = tf.Variable(tf.truncated_normal(shape = (3,3,1,64), mean = 0,stddev = 0.1))
        m_conv1_b = tf.Variable(tf.zeros(64))
        
        m_conv2_W = tf.Variable(tf.truncated_normal(shape = (3,3,64,64), mean = 0,stddev = 0.1))
        m_conv2_b = tf.Variable(tf.zeros(64))
        
        #m_conv3_W = tf.Variable(tf.truncated_normal(shape = (3,3,64,64),mean = 0, stddev = 0.1))
        #m_conv3_b = tf.Variable(tf.zeros(64))
        
        m_conv4_W = tf.Variable(tf.truncated_normal(shape = (3,3,64,128),mean = 0, stddev = 0.1))
        m_conv4_b = tf.Variable(tf.zeros(128))
        
        m_conv5_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,128),mean = 0,stddev = 0.1))
        m_conv5_b = tf.Variable(tf.zeros(128))
        
        #m_conv6_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,128),mean = 0, stddev = 0.1))
        #m_conv6_b = tf.Variable(tf.zeros(128))
        
        m_conv7_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,256), mean = 0,stddev = 0.1))
        m_conv7_b = tf.Variable(tf.zeros(256))
        
        m_conv8_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,256),mean = 0,stddev = 0.1))
        m_conv8_b = tf.Variable(tf.zeros(256))
        
        #m_conv9_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,256),mean = 0, stddev = 0.1))
        #m_conv9_b = tf.Variable(tf.zeros(256))
        
        m_conv10_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,512), mean = 0,stddev = 0.1))
        m_conv10_b = tf.Variable(tf.zeros(512))
        
        m_conv11_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,512), mean = 0,stddev = 0.1))
        m_conv11_b = tf.Variable(tf.zeros(512))
        
        #m_conv12_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,512),mean = 0, stddev = 0.1))
        #m_conv12_b = tf.Variable(tf.zeros(512))
        
        m_conv13_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,1024), mean = 0,stddev = 0.1))
        m_conv13_b = tf.Variable(tf.zeros(1024))
        
        m_conv14_W = tf.Variable(tf.truncated_normal(shape = (3,3,1024,1024), mean = 0,stddev = 0.1))
        m_conv14_b = tf.Variable(tf.zeros(1024))
        
        
        
        
        m_conv15_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,1024), mean = 0,stddev = 0.1))
        m_conv15_b = tf.Variable(tf.zeros(512))#
        
        m_conv16_W = tf.Variable(tf.truncated_normal(shape = (3,3,1024,512), mean = 0,stddev = 0.1))
        m_conv16_b = tf.Variable(tf.zeros(512))
        
        m_conv17_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,512),mean = 0,stddev = 0.1))
        m_conv17_b = tf.Variable(tf.zeros(512))
        
        m_conv18_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,512),mean = 0, stddev = 0.1))
        m_conv18_b = tf.Variable(tf.zeros(256))#
        
        m_conv19_W = tf.Variable(tf.truncated_normal(shape = (3,3,512,256), mean = 0, stddev = 0.1))
        m_conv19_b = tf.Variable(tf.zeros(256))
        
        m_conv20_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,256), mean = 0,stddev = 0.1))
        m_conv20_b = tf.Variable(tf.zeros(256))
        
        m_conv21_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,256), mean = 0,stddev = 0.1))
        m_conv21_b = tf.Variable(tf.zeros(128))#
        
        m_conv22_W = tf.Variable(tf.truncated_normal(shape = (3,3,256,128), mean = 0,stddev = 0.1))
        m_conv22_b = tf.Variable(tf.zeros(128))
        
        m_conv23_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,128),mean = 0,stddev = 0.1))
        m_conv23_b = tf.Variable(tf.zeros(128))
        
        m_conv24_W = tf.Variable(tf.truncated_normal(shape = (3,3,64,128),mean = 0, stddev = 0.1))
        m_conv24_b = tf.Variable(tf.zeros(64))
        
        m_conv25_W = tf.Variable(tf.truncated_normal(shape = (3,3,128,64), mean = 0,stddev = 0.1))
        m_conv25_b = tf.Variable(tf.zeros(64))
        
        m_conv26_W = tf.Variable(tf.truncated_normal(shape = (3,3,64,5), mean = 0,stddev = 0.1))
        m_conv26_b = tf.Variable(tf.zeros(5))
        
        m_conv1 = tf.nn.conv2d(self.x, m_conv1_W, strides = [1,1,1,1], padding = 'SAME') + m_conv1_b  #16,100,75
        #BN-Relu-Conv
        m_conv2_m, m_conv2_v = tf.nn.moments(m_conv1, axes = [0,1,2])
        m_conv2 = tf.nn.batch_normalization(m_conv1, mean = m_conv2_m, variance = m_conv2_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv2 = tf.nn.relu(m_conv1)
        m_conv2 = tf.nn.conv2d(m_conv2, m_conv2_W, strides = [1,1,1,1], padding = 'SAME') + m_conv2_b #16,100,75
        
        m_conv3 = tf.nn.relu(m_conv2)
        m_conv3 = tf.nn.max_pool(m_conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #16,50,38
        
        
        
        m_conv4_m, m_conv4_v = tf.nn.moments(m_conv3, axes = [0,1,2])
        m_conv4 = tf.nn.batch_normalization(m_conv3, mean = m_conv4_m, variance = m_conv4_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv4 = tf.nn.relu(m_conv4)
        m_conv4 = tf.nn.conv2d(m_conv4, m_conv4_W, strides = [1,1,1,1], padding = 'SAME') + m_conv4_b #32,50,38
        
        m_conv5_m, m_conv5_v = tf.nn.moments(m_conv4, axes = [0,1,2])
        m_conv5 = tf.nn.batch_normalization(m_conv4, mean = m_conv5_m, variance = m_conv5_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv5 = tf.nn.relu(m_conv5)
        m_conv5 = tf.nn.conv2d(m_conv5, m_conv5_W, strides = [1,1,1,1], padding = 'SAME') + m_conv5_b #32,50,38
        
        m_conv6 = tf.nn.relu(m_conv5)
        m_conv6 = tf.nn.max_pool(m_conv6, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #32,25,19
        
        
        
        m_conv7_m, m_conv7_v = tf.nn.moments(m_conv6, axes = [0,1,2])
        m_conv7 = tf.nn.batch_normalization(m_conv6, mean = m_conv7_m, variance = m_conv7_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv7 = tf.nn.relu(m_conv7)
        m_conv7 = tf.nn.conv2d(m_conv7, m_conv7_W, strides = [1,1,1,1], padding = 'SAME') + m_conv7_b #64,25,19
        
        m_conv8_m, m_conv8_v = tf.nn.moments(m_conv7, axes = [0,1,2])
        m_conv8 = tf.nn.batch_normalization(m_conv7, mean = m_conv8_m, variance = m_conv8_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv8 = tf.nn.relu(m_conv8)
        m_conv8 = tf.nn.conv2d(m_conv8, m_conv8_W, strides = [1,1,1,1], padding = 'SAME') + m_conv8_b #64,25,19
        
        m_conv9 = tf.nn.relu(m_conv8)
        m_conv9 = tf.nn.max_pool(m_conv9, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #64,50,38
        
        
        
        m_conv10_m, m_conv10_v = tf.nn.moments(m_conv9, axes = [0,1,2])
        m_conv10 = tf.nn.batch_normalization(m_conv9, mean = m_conv10_m, variance = m_conv10_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv10 = tf.nn.relu(m_conv10)
        m_conv10 = tf.nn.conv2d(m_conv10, m_conv10_W, strides = [1,1,1,1], padding = 'SAME') + m_conv10_b #128,50,38
        
        m_conv11_m, m_conv11_v = tf.nn.moments(m_conv10, axes = [0,1,2])
        m_conv11 = tf.nn.batch_normalization(m_conv10, mean = m_conv11_m, variance = m_conv11_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv11 = tf.nn.relu(m_conv11)
        m_conv11 = tf.nn.conv2d(m_conv11, m_conv11_W, strides = [1,1,1,1], padding = 'SAME') + m_conv11_b #128,50,38
        
        m_conv12 = tf.nn.relu(m_conv11)
        m_conv12 = tf.nn.max_pool(m_conv12, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #128,50,38
        
        
        
        
        m_conv13_m, m_conv13_v = tf.nn.moments(m_conv12, axes = [0,1,2])
        m_conv13 = tf.nn.batch_normalization(m_conv12, mean = m_conv13_m, variance = m_conv13_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv13 = tf.nn.relu(m_conv13)
        m_conv13 = tf.nn.conv2d(m_conv13, m_conv13_W, strides = [1,1,1,1], padding = 'SAME') + m_conv13_b #128,50,38
        
        m_conv14_m, m_conv14_v = tf.nn.moments(m_conv13, axes = [0,1,2])
        m_conv14 = tf.nn.batch_normalization(m_conv13, mean = m_conv14_m, variance = m_conv14_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv14 = tf.nn.relu(m_conv14)
        m_conv14 = tf.nn.conv2d(m_conv14, m_conv14_W, strides = [1,1,1,1], padding = 'SAME') + m_conv14_b #128,50,38
        
        
        
        
        m_conv15_m, m_conv15_v = tf.nn.moments(m_conv14, axes = [0,1,2])
        m_conv15 = tf.nn.batch_normalization(m_conv14, mean = m_conv15_m, variance = m_conv15_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv15 = tf.nn.relu(m_conv14)
        m_conv15 = tf.nn.conv2d_transpose(m_conv15, m_conv15_W, output_shape = [self.batch_,50,38,512], strides = [1,2,2,1], padding = 'SAME') + m_conv15_b
        m_conv15 = tf.concat([m_conv11,m_conv15],axis=3)
        
        m_conv16_m, m_conv16_v = tf.nn.moments(m_conv15, axes = [0,1,2])
        m_conv16 = tf.nn.batch_normalization(m_conv15, mean = m_conv16_m, variance = m_conv16_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv16 = tf.nn.relu(m_conv16)
        m_conv16 = tf.nn.conv2d(m_conv16, m_conv16_W, strides = [1,1,1,1], padding = 'SAME') + m_conv16_b
        
        m_conv17_m, m_conv17_v = tf.nn.moments(m_conv16, axes = [0,1,2])
        m_conv17 = tf.nn.batch_normalization(m_conv16, mean = m_conv17_m, variance = m_conv17_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv17 = tf.nn.relu(m_conv17)
        m_conv17 = tf.nn.conv2d(m_conv17, m_conv17_W, strides = [1,1,1,1], padding = 'SAME') + m_conv17_b
        
        
        
        
        m_conv18_m, m_conv18_v = tf.nn.moments(m_conv17, axes = [0,1,2])
        m_conv18 = tf.nn.batch_normalization(m_conv17, mean = m_conv18_m, variance = m_conv18_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv18 = tf.nn.relu(m_conv18)
        m_conv18 = tf.nn.conv2d_transpose(m_conv18, m_conv18_W, output_shape = [self.batch_,100,75,256], strides = [1,2,2,1], padding = 'SAME') + m_conv18_b
        m_conv18 = tf.concat([m_conv8,m_conv18],axis=3)
        
        m_conv19_m, m_conv19_v = tf.nn.moments(m_conv18, axes = [0,1,2])
        m_conv19 = tf.nn.batch_normalization(m_conv18, mean = m_conv19_m, variance = m_conv19_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv19 = tf.nn.relu(m_conv19)
        m_conv19 = tf.nn.conv2d(m_conv19, m_conv19_W, strides = [1,1,1,1], padding = 'SAME') + m_conv19_b
        
        m_conv20_m, m_conv20_v = tf.nn.moments(m_conv19, axes = [0,1,2])
        m_conv20 = tf.nn.batch_normalization(m_conv19, mean = m_conv20_m, variance = m_conv20_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv20 = tf.nn.relu(m_conv20)
        m_conv20 = tf.nn.conv2d(m_conv20, m_conv20_W, strides = [1,1,1,1], padding = 'SAME') + m_conv20_b
        
        
        
        
        m_conv21_m, m_conv21_v = tf.nn.moments(m_conv20, axes = [0,1,2])
        m_conv21 = tf.nn.batch_normalization(m_conv20, mean = m_conv21_m, variance = m_conv21_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv21 = tf.nn.relu(m_conv21)
        m_conv21 = tf.nn.conv2d_transpose(m_conv21, m_conv21_W, output_shape = [self.batch_,200,150,128], strides = [1,2,2,1], padding = 'SAME') + m_conv21_b
        m_conv21 = tf.concat([m_conv5,m_conv21],axis=3)
        
        m_conv22_m, m_conv22_v = tf.nn.moments(m_conv21, axes = [0,1,2])
        m_conv22 = tf.nn.batch_normalization(m_conv21, mean = m_conv22_m, variance = m_conv22_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv22 = tf.nn.relu(m_conv22)
        m_conv22 = tf.nn.conv2d(m_conv22, m_conv22_W, strides = [1,1,1,1], padding = 'SAME') + m_conv22_b
        
        m_conv23_m, m_conv23_v = tf.nn.moments(m_conv22, axes = [0,1,2])
        m_conv23 = tf.nn.batch_normalization(m_conv22, mean = m_conv23_m, variance = m_conv23_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv23 = tf.nn.relu(m_conv23)
        m_conv23 = tf.nn.conv2d(m_conv23, m_conv23_W, strides = [1,1,1,1], padding = 'SAME') + m_conv23_b
        
        
        
        m_conv24_m, m_conv24_v = tf.nn.moments(m_conv23, axes = [0,1,2])
        m_conv24 = tf.nn.batch_normalization(m_conv23, mean = m_conv24_m, variance = m_conv24_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv24 = tf.nn.relu(m_conv24)
        m_conv24 = tf.nn.conv2d_transpose(m_conv24, m_conv24_W, output_shape = [self.batch_,400,300,64], strides = [1,2,2,1], padding = 'SAME') + m_conv24_b
        m_conv24 = tf.concat([m_conv2,m_conv24],axis=3)
        
        m_conv25_m, m_conv25_v = tf.nn.moments(m_conv24, axes = [0,1,2])
        m_conv25 = tf.nn.batch_normalization(m_conv24, mean = m_conv25_m, variance = m_conv25_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv25 = tf.nn.relu(m_conv25)
        m_conv25 = tf.nn.conv2d(m_conv25, m_conv25_W, strides = [1,1,1,1], padding = 'SAME') + m_conv25_b
        
        m_conv26_m, m_conv26_v = tf.nn.moments(m_conv25, axes = [0,1,2])
        m_conv26 = tf.nn.batch_normalization(m_conv25, mean = m_conv26_m, variance = m_conv26_v, offset = 0, scale = 1, variance_epsilon = 0.01)
        m_conv26 = tf.nn.relu(m_conv26)
        m_conv26 = tf.nn.conv2d(m_conv26, m_conv26_W, strides = [1,1,1,1], padding = 'SAME') + m_conv26_b
        
        
        y = m_conv26
    
        return tf.nn.softmax(y)
    
    
    


