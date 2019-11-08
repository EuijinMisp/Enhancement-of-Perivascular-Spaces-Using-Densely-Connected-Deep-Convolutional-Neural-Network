# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:36:04 2018

@author: Euijin Jung , dmlwls3876@dgist.ac.kr

"""

import tensorflow as tf
import math
import time
import utils
import os
import numpy as np


class model_set:
    
    def __init__(self,img_patch_size,generator,learn_rate,batch_size,decay_step):

        self. img_patch_size=img_patch_size
        
        self.x_input = tf.placeholder(tf.float32, [None,self.img_patch_size,self.img_patch_size,self.img_patch_size, 1], name='images')
        
        self.y_ = tf.placeholder(tf.float32, [None,self.img_patch_size,self.img_patch_size,self.img_patch_size, 1], name='labels')
       
        self.rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        self.batch = tf.placeholder(tf.int64, shape=[], name='batch')   

        self.y_pred = generator(self.x_input) 

        self.learn_rate =learn_rate             
        
        self.batch_size = batch_size      

        self.decay_step =decay_step  
        
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y_pred))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.rate)
        
        self.model_train = self.optimizer.minimize(self.loss)
        
        self.saver = tf.train.Saver(max_to_keep=50)
        
        self.init = tf.global_variables_initializer()  
    
    def training(self,data,load_model,save_model,epochs,summary_path,npatient,mode):

        self.widthcut = ["229.img",
                  "230.img",
                  "232.img",
                  "1203.img",
                  "242.img",
                  "243.img",
                  "244.img",
                  "245.img",
                  "250.img",
                  "251.img",
                  "261.img"]

        with tf.Session() as sess:
            start = time.time()
            loss_list=[]
            epoch_list=[]
            gap=10
            h_gap=int(gap/2)

            if os.path.isfile(summary_path):
                
                summary = open(summary_path,'a')
                summary.write('\n')             
            else:
                summary = open(summary_path,'w')
                
            init_learnrate = np.copy(self.learn_rate)
            summary.write('initial_learning_rate:{}\n'.format(self.learn_rate))
            summary.write('\n')
            try:
                self.saver.restore(sess, load_model)
                print('Initialization loaded')
        
            except:
                sess.run(self.init)
                print('New initialization done')
            
           
            for n in range(1,epochs+1):                
                acum_loss = 0

                for m in range(0, data[0].shape[0], self.batch_size):
                    m2 = min(data[0].shape[0], m+self.batch_size)
                    _, result,pred= sess.run((self.model_train, self.loss,self.y_pred), feed_dict = {self.x_input: data[0][m:m2],\
                                                   self.y_: data[1][m:m2],self.rate:self.learn_rate, self.batch: m2-m})
                    
                    acum_loss += result*(m2-m)

                if (n%5)==0 or n==1: 

                    print("learning_rate :{}".format(self.learn_rate))
                    print('epoch: {}'.format(n))
                    print('train_loss = {}'.format(acum_loss)) 
                    summary.write('learning_rate: {}\n'.format(self.learn_rate))
                    summary.write('epoch: {}\n'.format(n))
                    summary.write('train_loss = {}\n'.format(acum_loss))
                    summary.write('\n')
                   
                    epoch_list.append(n)
                    loss_list.append(acum_loss)
                    
                    self.saver.save(sess, save_model,global_step=n)
                    self.saver.save(sess, save_model)
                    print("model_saved")
                    print()
                
                if (n%self.decay_step)==0:
                    self.learn_rate = self.learn_rate-(init_learnrate/epochs)
                
            self.saver.save(sess, save_model)
            end = time.time()
            
            hour_time = int((end-start)//3600)
            min_time = int(((end-start)-hour_time*3600)//60)
            sec_time = int(((end-start)-hour_time*3600-min_time*60)%60)
            
            print("model_saved")
            print("training time : %d hour, %d min, %d sec"%(hour_time,min_time,sec_time))
            summary.write("training time : %d hour, %d min, %d sec\n"%(hour_time,min_time,sec_time))
            summary.write("\n")            
            summary.close()

        print('Training done!!')       
        return epoch_list,loss_list
    
        
     
    def testing( self, img_test_data, load_model):
        
        with tf.Session() as sess:
            try:           
                self.saver.restore(sess, load_model)
                print('Initialization loaded')
        
            except:
                sess.run(self.init)
                print('New initialization done')
                
            pred_list = []
       
            for m in range(0, img_test_data.shape[0], self.batch_size):               
                m2 = min(img_test_data.shape[0], m+self.batch_size)
    
                prediction = sess.run((self.y_pred), feed_dict = {self.x_input: img_test_data[m:m2], self.batch: m2-m})    
                pred_list = pred_list + list(prediction)
         
        print("Test is done")
        return pred_list

    
    
    
    
    
    
    
    
    
    
    
    
    
