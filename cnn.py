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


class DCNN6_SC_B(object):

    def __init__(self,
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
        self.filter_size = filter_size

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

            nextlayer =conv_in[-1]
            conv.append(conv_in)
        return conv

    def bot_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.bot_weight, strides=[1,1,1,1,1], padding='SAME') + self.bot_biases                              
        x = tf.nn.relu(x)
        return x 

    def reconv_layer(self, input_layer):
        x = tf.nn.conv3d(input_layer, self.reconv_weight, strides=[1,1,1,1,1], padding='SAME') + self.reconv_biases
        x = tf.nn.tanh(x)
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


    def build_generator(self,images):
        
        self.images = images      
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