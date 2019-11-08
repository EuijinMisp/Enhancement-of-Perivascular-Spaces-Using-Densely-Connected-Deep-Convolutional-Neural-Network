# -*- coding: utf-8 -*-
"""
Created on Tue April 10 10:06:01 2018
@author: Euijin Jung

"""

import tensorflow as tf
import os
import numpy as np
import config
import pandas as pd
import time
import math
import cnn
import learn 
import utils
import SimpleITK as sitk

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

current_dir = FLAGS.current_dir

current_GPU = FLAGS.current_GPU

os.chdir(current_dir)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(current_GPU)

mode = FLAGS.mode

pvs_path = FLAGS.pvs_path

testout_path = FLAGS.testout_path

train_path = FLAGS.train_path

load_modelpath = train_path+FLAGS.load_model

save_modelpath = train_path+FLAGS.save_model

summary_path = load_modelpath + FLAGS.summary_Name

img_path = FLAGS.image_dir

label_path = FLAGS.label_dir

seg_path = FLAGS.seg_dir

img_type = FLAGS.image_type

npatient = FLAGS.npatient

batch_size = FLAGS.batchsize

learning_rate = FLAGS.learningRate

leaning_epochs= FLAGS.epoch

decay_step = FLAGS.decay_step

fs = FLAGS.fs

img_patch_size = FLAGS.img_patchsize

train_img_pat_st = FLAGS.train_img_pat_st

n_dbH = FLAGS.n_dbH

n_dbA = FLAGS.n_dbA

def main(argv=None):
    
    data = utils.Medical_data(img_path,label_path,seg_path,pvs_path,train_img_pat_st,img_patch_size)  
    
    dataset = list(data.dataload(npatient,img_type,mode))# data load          
    print("Total input shape:{}".format(dataset[0].shape))
                
    gap = img_patch_size -train_img_pat_st
    h_gap = int(gap/2)

    network = cnn.DCNN6_SC_B(c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
    network_name="DenseNet"
    
    print("nework : DenseNet" )
    Generator= network.build_generator 
    model = learn.model_set(img_patch_size,Generator,learning_rate,batch_size,decay_step)

    if mode =="train":
        try:
            os.makedirs(train_path)
        except OSError:
            pass

        print("mode: train")
        print('load model:%s'%load_modelpath)
        
        epoch,loss= model.training(dataset,load_modelpath, save_modelpath,leaning_epochs,summary_path,npatient,mode) #training

        df = pd.DataFrame({"epoch":epoch,"loss":loss})
            
        writer = pd.ExcelWriter(save_modelpath+"_.xlsx", engine='xlsxwriter')
        
        df.to_excel(writer, sheet_name='Sheet1')
        
        writer.save()    

        print("save excel done")
              
    elif mode =="test":
        try:
            os.makedirs(testout_path)
        except OSError:
            pass
      
        start = time.time()
        print("mode: test")
        print('load model path :%s'%load_modelpath)
        files = os.listdir(img_path)
        file_name=[]
        
        test_xy = model.testing(dataset[0],load_modelpath) #testing
        test_xy = np.array(test_xy)
        test_xy = test_xy[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
        dataset[0] = dataset[0][:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
        dataset[1] = dataset[1][:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
        recon_xy = data.recon_3d(test_xy,npatient) ##reconstruction
        dataset[1] = data.recon_3d(dataset[1],npatient)
        dataset[0] = data.recon_3d(dataset[0],npatient)

        for i in range(len(files)):
            if files[i][-len(img_type):]==img_type:
                file_name.append(files[i])
            
        PVS_psnr_list_xy=[]
        PVS_ssim_list_xy=[]
        WM_psnr_list_xy=[]
        WM_ssim_list_xy=[]

        avg_psnr_PVS_xy=0
        avg_ssim_PVS_xy=0
        avg_psnr_WM_xy=0
        avg_ssim_WM_xy=0

        name=[]
        for i in range(npatient):
            f_name = file_name[i]
            brain_mask = np.copy(dataset[2][i]) # dataset[2] : labeled form 0~3 , dataset[5]: all brain label
            brain_mask[brain_mask==2]=1
            brain_mask[brain_mask==3]=0
            brain_img = dataset[5][i] * brain_mask 
            
            recon_xy[i] = brain_img + (recon_xy[i] * dataset[3][i])
            dataset[1][i] = brain_img + (dataset[1][i] * dataset[3][i])            
            dataset[0][i] = brain_img + (dataset[0][i] * dataset[3][i])
                
            img_file = sitk.ReadImage(os.path.join(img_path, f_name ))
            utils.save_3dimg (testout_path,"%s_pred_xy.nii"%f_name,img_file,recon_xy[i])
            utils.save_3dimg (testout_path,"%s_y.nii"%f_name,img_file,dataset[1][i])
            utils.save_3dimg (testout_path,"%s_x.nii"%f_name,img_file,dataset[0][i])
            
            wm_pred_xy = recon_xy[i] * dataset[3][i]
            wm_y = dataset[1][i] * dataset[3][i] 
            wm_x = dataset[0][i] * dataset[3][i]
            
            WM_psnr_xy = utils.psnr(wm_y, wm_pred_xy)
            WM_ssim_xy = utils.ssim3d(wm_y,wm_pred_xy)            
            pvs_pred_xy = recon_xy[i] * dataset[4][i]
            pvs_y = dataset[1][i] * dataset[4][i]               
                   
            pvs_psnr_xy = utils.psnr(pvs_y, pvs_pred_xy)
            pvs_ssim_xy = utils.ssim3d(pvs_y, pvs_pred_xy)

            avg_psnr_PVS_xy+=pvs_psnr_xy
            avg_ssim_PVS_xy+=pvs_ssim_xy
            avg_psnr_WM_xy+=WM_psnr_xy
            avg_ssim_WM_xy+=WM_ssim_xy
                      
            PVS_psnr_list_xy.append(pvs_psnr_xy)
            PVS_ssim_list_xy.append(pvs_ssim_xy)
            WM_psnr_list_xy.append(WM_psnr_xy)
            WM_ssim_list_xy.append(WM_ssim_xy)           
            name.append(f_name+"(WM)")
            

            print("patient: %s"%f_name)
            print("PVS_psnr = {:.5f}".format(pvs_psnr_xy))
            print("PVS_ssim = {:.5f}".format(pvs_ssim_xy))
            print("WM_psnr = {:.5f}".format(WM_psnr_xy))
            print("WM_ssim = {:.5f}".format(WM_ssim_xy))  
            print(" ")
            
        
        avg_psnr_PVS_xy/=npatient
        avg_ssim_PVS_xy/=npatient
        avg_psnr_WM_xy/=npatient
        avg_ssim_WM_xy/=npatient
        
        name.append("average")
        PVS_psnr_list_xy.append(avg_psnr_PVS_xy)
        PVS_ssim_list_xy.append(avg_ssim_PVS_xy)
        WM_psnr_list_xy.append(avg_psnr_WM_xy)
        WM_ssim_list_xy.append(avg_ssim_WM_xy)
       
    
        print("avg_psnr_PVS = {:.5f}".format(avg_psnr_PVS_xy))
        print("avg_ssim_PVS = {:.5f}".format(avg_ssim_PVS_xy))
        print("avg_psnr_WM = {:.5f}".format(avg_psnr_WM_xy))
        print("avg_ssim_WM = {:.5f}".format(avg_ssim_WM_xy))
        print(" ")
      
        df = pd.DataFrame({"patient":name,"PVS_psnr_xy": PVS_psnr_list_xy,"WM_psnr_xy":WM_psnr_list_xy,"WM_ssim_xy":WM_ssim_list_xy,\
                           "PVS_ssim_xy":PVS_ssim_list_xy})
            
        writer = pd.ExcelWriter(testout_path+"/"+network_name+load_modelpath[-3:]+".xlsx", engine='xlsxwriter')       
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        print("save excel done")
        
        end = time.time()          
        hour_time = int((end-start)//3600)
        min_time = int(((end-start)-hour_time*3600)//60)
        sec_time = int(((end-start)-hour_time*3600-min_time*60)%60)
        print("training time : %d hour, %d min, %d sec"%(hour_time,min_time,sec_time))
        
    
if __name__ == '__main__':
  tf.app.run()

