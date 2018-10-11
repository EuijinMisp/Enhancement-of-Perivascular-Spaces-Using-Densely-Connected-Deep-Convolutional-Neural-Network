# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:36:04 2018

@author: Euijin Jung , dmlwls3876@dgist.ac.kr

"""

#import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import time
import utils
import os
import numpy as np
#import SimpleITK as sitk

def recon_3d(test_out,label_shape,gap,img_patch_st):
        
    cnt=0
    z = label_shape[0]
    y = label_shape[1]       
    x = label_shape[2]   
    recon_out = np.zeros((z,y,x))

    stich = img_patch_st   
    step0 = int(z/stich) 
    step1 = int(y/stich) 
    step2 = int(x/stich) 
    
    for i in range(step0):
        for j in range(step1):
            for k in range(step2):
                recon_out[stich*i:stich*i+stich, stich*j:stich*j+stich, stich*k:stich*k+stich] = test_out[cnt]
                cnt+=1 
    return recon_out

class learning_model:
    
    def __init__(self,x_input,y_,y_pred,rate,batch,batch_size):
        
        self.x_input = x_input
        
        self.y_ = y_
        
        self.y_pred = y_pred
       
        self.rate = rate
        
        self.batch = batch                  
        
        self.batch_size = batch_size        
        
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y_pred))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.rate)
        
        self.model_train = self.optimizer.minimize(self.loss)
        
        #self.y_enc = tf.round(self.y_pred)
        
        #self.acc = tf.reduce_sum(tf.squared_difference(self.y_, self.y_enc))
        
        #self.total = tf.reduce_prod(tf.shape(self.y_enc))
        
        self.saver = tf.train.Saver(max_to_keep=50)
        
        self.init = tf.global_variables_initializer()  
    
    def training(self,data,load_model,save_model,learn_rate,epochs,decay_step,factor,\
        summary_path,n_val,val_img_path,val_label_path,val_seg_path,val_pvsseg_path,img_patch_size,img_type,npatient,mode):

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
            loss_test_list=[]
            epoch_list=[]
            wm_psnr_list=[]
            wm_ssim_list=[]
            pvs_psnr_list=[]
            pvs_ssim_list=[]
            gap=10
            h_gap=int(gap/2)


            if os.path.isfile(summary_path):
                
                summary = open(summary_path,'a')
                summary.write('\n')             
            else:
                summary = open(summary_path,'w')
                
            init_learnrate = learn_rate
            summary.write('initial_learning_rate:{}\n'.format(learn_rate))
            summary.write('\n')
            try:
                self.saver.restore(sess, load_model)
                print('Initialization loaded')
        
            except:
                sess.run(self.init)
                print('New initialization done')
            
            
            for n in range(1,epochs+1):                
                acum_loss = 0
                img_train_data, label_train_data= data.dataload(npatient,img_type,mode) # data load

                for m in range(0, img_train_data.shape[0], self.batch_size):
                    m2 = min(img_train_data.shape[0], m+self.batch_size)
                    _, result,pred= sess.run((self.model_train, self.loss,self.y_pred), feed_dict = {self.x_input: img_train_data[m:m2],\
                                                   self.y_: label_train_data[m:m2],self.rate:learn_rate, self.batch: m2-m})
                    
                    acum_loss += result*(m2-m)
                del img_train_data, label_train_data                  

                if (n%10)==0 or n==1: 
                    """
                    acum_test_loss=0
                    wm_psnr=0
                    wm_ssim=0
                    pvs_psnr=0
                    pvs_ssim=0
                    data_val = utils.Medical_data(val_img_path,val_label_path,val_seg_path,val_pvsseg_path,50,img_patch_size)  
                    
                    for i in range(n_val):
                        val_img,val_label,val_wm,val_pvs= data_val.dataload(1,img_type,"validation",val_order=i) # validation data load        
                        pred_list=[]
                        label_list=[]
                        for m in range(0, val_img.shape[0], self.batch_size):#validation
                            m2 = min(val_img.shape[0], m+self.batch_size)
                            pred,result= sess.run((self.y_pred,self.loss), feed_dict = {self.x_input: val_img[m:m2],\
                                                           self.y_: val_label[m:m2], self.batch: m2-m})                                            
                            
                            acum_test_loss += result*(m2-m)                       
                            pred_list = pred_list + list(pred[:,h_gap:pred.shape[1]-h_gap,h_gap:pred.shape[2]-h_gap,h_gap:pred.shape[3]-h_gap,0])
                            label_list = label_list + list(val_label[m:m2,h_gap:pred.shape[1]-h_gap,h_gap:pred.shape[2]-h_gap,h_gap:pred.shape[3]-h_gap,0])
    
                        del val_img,val_label,pred
                        
                        pred_list = np.array(pred_list)
                        label_list = np.array(label_list)
                        
                        pred_list = recon_3d(pred_list,val_wm.shape,gap,pred_list.shape[1])
                        label_list = recon_3d(label_list,val_wm.shape,gap,label_list.shape[1])
                                                     
                        #img_file = sitk.ReadImage(os.path.join("D:/DGIST/Research/server_PVS/data/set2/image","242.img"))
                        #utils.save_3dimg("D:/DGIST/Research/server_PVS","test%d.nii"%i,img_file,pred_list)
                        #utils.save_3dimg("D:/DGIST/Research/server_PVS","label%d.nii"%i,img_file,label_list)
                        
                        wm_pred = pred_list*val_wm #WM measuring
                        wm_label = label_list*val_wm                  
    
                        wm_psnr += utils.psnr(wm_pred,wm_label)
                        wm_ssim += utils.ssim3d(wm_pred, wm_label)
                                               
                        pvs_pred = pred_list*val_pvs #WM measuring
                        pvs_label = label_list*val_pvs                  
    
                        pvs_psnr += utils.psnr(pvs_pred,pvs_label)
                        pvs_ssim += utils.ssim3d(pvs_pred, pvs_label)
                        
                        del val_wm,val_pvs,pred_list,label_list             
                        
                    wm_psnr/=n_val                       
                    wm_ssim/=n_val
                    pvs_psnr/=n_val                       
                    pvs_ssim/=n_val
                    """
                    print("learning_rate :{}".format(learn_rate))
                    print('epoch: {}'.format(n))
                    print('train_loss = {}'.format(acum_loss)) 
                    #print('test_loss = {}'.format(acum_test_loss))                    
                    #print('test_wm_psnr ={}'.format(wm_psnr))
                    #print('test_wm_ssim ={}'.format(wm_ssim))
                    #print('test_pvs_psnr ={}'.format(pvs_psnr))
                    #print('test_pvs_ssim ={}'.format(pvs_ssim))

                    print()
                    summary.write('learning_rate: {}\n'.format(learn_rate))
                    summary.write('epoch: {}\n'.format(n))
                    summary.write('train_loss = {}\n'.format(acum_loss))
                    #summary.write('test_loss = {}\n'.format(acum_test_loss))
                    #summary.write('test_wm_psnr ={}\n'.format(wm_psnr))
                    #summary.write('test_wm_ssim ={}\n'.format(wm_ssim))
                    #summary.write('test_pvs_psnr ={}\n'.format(pvs_psnr))
                    #summary.write('test_pvs_ssim ={}\n'.format(pvs_ssim))
    
                    summary.write('\n')
                    
                    epoch_list.append(n)
                    loss_list.append(acum_loss)
                    #loss_test_list.append(acum_test_loss)
                    #wm_psnr_list.append(wm_psnr)
                    #wm_ssim_list.append(wm_ssim)
                    #pvs_psnr_list.append(pvs_psnr)
                    #pvs_ssim_list.append(pvs_ssim)
                    
                    self.saver.save(sess, save_model,global_step=n)
                    self.saver.save(sess, save_model)
                    print("model_saved")
                    print()
                
                if (n%decay_step)==0:
                    #learn_rate = learn_rate/factor
                    learn_rate = learn_rate-(init_learnrate*factor)
                
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
        print("")            
        print('Training done!!')
        print("")        
        return epoch_list,loss_list#,loss_test_list,wm_psnr_list,wm_ssim_list,pvs_psnr_list,pvs_ssim_list
    
        
     
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
    
    """
    def dsc_eval(self,result, seg_img, class_num):
        
        dsc_list =[]
        for i in range(class_num):
        
            dsc_img = np.copy(seg_img)
            dsc_img2 = np.copy(result)
            
            dsc_img[dsc_img > i] = 0
            dsc_img[dsc_img < i] = 0
            dsc_img2[dsc_img2 > i] = 0
            dsc_img2[dsc_img2 < i] = 0
            
            dsc_img[dsc_img==i] = 1
            dsc_img2[dsc_img2==i] = 1
            
            numer = dsc_img+dsc_img2
            numer[numer==1]=0
            numer = np.sum(numer)
            denom = np.sum(dsc_img+dsc_img2)
            dsc = numer / denom            
            dsc_list.append(dsc)
                                        
        return dsc_list
    """
