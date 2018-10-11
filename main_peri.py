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
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

current_dir = FLAGS.current_dir

current_GPU = FLAGS.current_GPU


os.chdir(current_dir)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(current_GPU)

import cnn
from learn import learning_model
import utils
import SimpleITK as sitk

mode = FLAGS.mode

val_img_path = FLAGS.val_img_path

val_label_path= FLAGS.val_label_path

val_seg_path = FLAGS.val_seg_path

val_pvs_path = FLAGS.val_pvs_path

pvs_path = FLAGS.pvs_path

testout_path = FLAGS.testout_path

network_num = FLAGS.network_num

load_modelpath = FLAGS.model_dir

save_modelpath = FLAGS.save_dir

summary_path = load_modelpath + FLAGS.summary_Name

img_path = FLAGS.image_dir

label_path = FLAGS.label_dir

seg_path = FLAGS.seg_dir

img_type = FLAGS.image_type

npatient = FLAGS.npatient

n_patient_val = FLAGS.n_patient_val

batch_size = FLAGS.batchsize

learning_rate = FLAGS.learningRate

leaning_epochs= FLAGS.epoch

decay_step = FLAGS.decay_step

factor =  FLAGS.factor

fs = FLAGS.fs

img_patch_size = FLAGS.img_patchsize

train_img_pat_st = FLAGS.train_img_pat_st

n_dbH = FLAGS.n_dbH

n_dbA = FLAGS.n_dbA


def main(argv=None):
    
    ###data load###
    
    data = utils.Medical_data(img_path,label_path,seg_path,pvs_path,train_img_pat_st,img_patch_size)    
  
    ### model in-out define ###
      
    batch = tf.placeholder(tf.int32, shape=[], name='batch')
    
    rate = tf.placeholder(tf.float32)
    
    gap = img_patch_size -train_img_pat_st
    
    h_gap = int(gap/2)

    image = tf.placeholder(tf.float32, [None,img_patch_size,img_patch_size,img_patch_size, 1], name='images')
    
    label = tf.placeholder(tf.float32, [None,img_patch_size,img_patch_size,img_patch_size, 1], name='labels')
    
    if network_num == "0": # DCNN_SC_B
        
        network = cnn.DCNN6_SC_B(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        
        network_name="DCNN_SC_B"
    
        print("nework : DCNN_SC_B" )
    
    elif network_num == "1": # DCNN_2d
                            
        network = cnn.RDensenet_Symskip2(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="RDensenet_symskip2"
    
        print("nework : RDensenet_symskip2" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))
        
    elif network_num == "2":# 3d SRCNN
                   
        network = cnn.SRCNN(image)
        
        dimension=3
        
        network_name="SRCNN"
       
        print("nework : SRCNN")

    
    elif network_num == "3":# 2d SRCNN

        image = tf.placeholder(tf.float32, [None,img_patch_size,img_patch_size, 1], name='images')
    
        label = tf.placeholder(tf.float32, [None,img_patch_size,img_patch_size, 1], name='labels')
                      
        network = cnn.SRCNN2d(image)
        
        #x_low= np.reshape(x_low,(x_low.shape[0]*x_low.shape[1],img_patch_size,img_patch_size,1))
        #y_origin= np.reshape(y_origin,(y_origin.shape[0]*y_origin.shape[1],img_patch_size,img_patch_size,1))

        network_name="SRCNN_2d"
        
        dimension=2
        
        print("nework : SRCNN_2d")
            
    elif network_num == "4": # DCNN
                   
        network = cnn.DCNN(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="DCNN"
        
        print("nework : DCNN" )


    elif network_num == "5": # DCNN_SC
           
        network = cnn.DCNN_SC(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="DCNN_SC"
        
        print("nework : DCNN_SC" )    

    elif network_num == "6": # VDSR
           
        network = cnn.VDSR(image,num_layers=n_dbA)
        
        dimension=3
        network_name="VDSR"
        
        print("nework : VDSR" )


    elif network_num == "7": # EDSR
           
        network = cnn.EDSR(image,num_blocks=n_dbA,num_layers=n_dbH)
        
        dimension=3
        network_name="EDSR"
        
        print("nework : EDSR" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))

    elif network_num == "8": # RDensenet
           
        network = cnn.RDensenet(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="RDensenet"
    
        print("nework : RDensenet" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))
    elif network_num == "9": # RDensenet
           
        network = cnn.RDensenet_Symskip(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="RDensenet_symskip"
    
        print("nework : RDensenet_symskip" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))

    elif network_num == "11": # DDCNN_SC_B
           
        network = cnn.DDCNN_SC_B(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        network_name="DDCNN_SC_B"
        
        dimension=3
    
        print("nework : DDCNN_SC_B")
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))
    elif network_num == "12": # DDCNN_SC
           
        network = cnn.DDCNN_SC(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="DDCNN_SC"
    
        print("nework : DDCNN_SC" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))
    elif network_num == "13": # DDCNN

        network = cnn.DDCNN(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="DDCNN"
    
        print("nework : DDCNN" )
        print("nblock:" + str(n_dbA))
        print("nlayer:" + str(n_dbH))

    elif network_num == "14": # DCNN1

        network = cnn.RDDCNN_SC_B(image,c_dim =1,des_block_H=n_dbH,des_block_ALL=n_dbA,growth_rate=8,filter_size=fs)
        
        dimension=3
        network_name="RDDCNN_SC_B"
    
        print("nework : RDDCNN_SC_B" )

   
    prediction = network.build_model()
    
    model = learning_model(image, label, prediction, rate, batch, batch_size)
    
    
    if mode =="train":
        
        print("mode: train")
        print('load model:%s'%load_modelpath)
    
        epoch,train_loss= model.training(data,load_modelpath, save_modelpath,\
        learning_rate,leaning_epochs,decay_step,factor,summary_path,n_patient_val,val_img_path,val_label_path,val_seg_path,val_pvs_path,img_patch_size,img_type,npatient,mode="train") #training
        
        df = pd.DataFrame({"epoch":epoch,"train loss":train_loss})
            
        writer = pd.ExcelWriter(save_modelpath+"_.xlsx", engine='xlsxwriter')
        
        df.to_excel(writer, sheet_name='Sheet1')
        
        writer.save()
        
        print("save excel done")
    
    
        
    elif mode =="test":
        x_low, y_origin,seg_list,WM_list,PVS_list,label_brain= data.dataload(npatient,img_type,mode) # data load  
        print(" Test Data load done")  
        print("mode: test")
        print('load model path :%s'%load_modelpath)
        
        test_out = model.testing(x_low, load_modelpath) #testing
        
        test_out = np.array(test_out)
                
        if dimension == 3: #srdense 3d or srcnn3d
            print("dimension = %d"%dimension)
            test_out = test_out[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
            x_low = x_low[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
            y_origin = y_origin[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
            
    
        elif dimension == 2: #srdense 2d or srcnn2d
            print("dimension = %d"%dimension)
    
            test_out = np.reshape(test_out,(int(len(test_out)/img_patch_size),img_patch_size,img_patch_size,img_patch_size,1))
            x_low = np.reshape(x_low,(int(len(x_low)/img_patch_size),img_patch_size,img_patch_size,img_patch_size,1))
            y_origin = np.reshape(y_origin,(int(len(y_origin)/img_patch_size),img_patch_size,img_patch_size,img_patch_size,1))
            
            test_out = test_out[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
            x_low = x_low[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
            y_origin = y_origin[:,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,h_gap:img_patch_size-h_gap,0]
     
               
        recon_out = data.recon_3d(test_out,npatient) ##reconstruction
        y_origin = data.recon_3d(y_origin,npatient)
        x_low = data.recon_3d(x_low,npatient)
        
        files = os.listdir(img_path)
        file_name=[]
        for i in range(len(files)):
            if files[i][-len(img_type):]==img_type:
                file_name.append(files[i])
            
        PVS_psnr_list=[]
        PVS_ssim_list=[]
        WM_psnr_list=[]
        WM_ssim_list=[]
        avg_psnr_PVS=0
        avg_ssim_PVS=0
        avg_psnr_WM=0
        avg_ssim_WM=0
        name=[]
        for i in range(npatient):
            f_name = file_name[i]
            brain_mask = np.copy(seg_list[i]) # seg_list : labeled form 0~3 , label_brain: all brain label
            brain_mask[brain_mask==2]=1
            brain_mask[brain_mask==3]=0
            brain_img = label_brain[i] * brain_mask 
            
            recon_out[i] = brain_img + (recon_out[i] * WM_list[i])
            y_origin[i] = brain_img + (y_origin[i] * WM_list[i])            
            x_low[i] = brain_img + (x_low[i] * WM_list[i])
                
            #img_file = sitk.ReadImage(os.path.join(img_path, f_name ))
            #utils.save_3dimg (testout_path,"%s_pred.nii"%f_name,img_file,recon_out[i])
            #utils.save_3dimg (testout_path,"%s_y.nii"%f_name,img_file,y_origin[i])
            #utils.save_3dimg (testout_path,"%s_x.nii"%f_name,img_file,x_low[i])
            
            wm_pred = recon_out[i] * WM_list[i]
            wm_y = y_origin[i] * WM_list[i]            
            wm_x = x_low[i] * WM_list[i]
            
            #utils.save_3dimg (testout_path,"%s_pred_WM.nii"%f_name,img_file,wm_pred)
            #utils.save_3dimg (testout_path,"%s_y_WM.nii"%f_name,img_file,wm_y)
            #utils.save_3dimg (testout_path,"%s_x_WM.nii"%f_name,img_file,wm_x)
    
            WM_psnr = utils.psnr(wm_y, wm_pred)
            WM_ssim = utils.ssim3d(wm_y,wm_pred)
            
            pvs_pred = recon_out[i] * PVS_list[i]
            pvs_y = y_origin[i] * PVS_list[i]            
            #utils.save_3dimg (testout_path,"%s_wm_mask.nii"%f_name,img_file,WM_list[i])
            #utils.save_3dimg (testout_path,"%s_pvs_mask.nii"%f_name,img_file,PVS_list[i])
            #pvs_x = x_low[i] * PVS_list[i]            
            
            pvs_psnr = utils.psnr(pvs_y, pvs_pred)
            pvs_ssim = utils.ssim3d(pvs_y, pvs_pred)
            
            
            avg_psnr_PVS+=pvs_psnr
            avg_ssim_PVS+=pvs_ssim
            avg_psnr_WM+=WM_psnr
            avg_ssim_WM+=WM_ssim
            
            PVS_psnr_list.append(pvs_psnr)
            PVS_ssim_list.append(pvs_ssim)
            WM_psnr_list.append(WM_psnr)
            WM_ssim_list.append(WM_ssim)
            name.append(f_name+"(WM)")
            
            print("patient: %s"%f_name)
            print("PVS_psnr = {:.5f}".format(pvs_psnr))
            print("PVS_ssim = {:.5f}".format(pvs_ssim))
            print("WM_psnr = {:.5f}".format(WM_psnr))
            print("WM_ssim = {:.5f}".format(WM_ssim))          
            print(" ")
        
        avg_psnr_PVS/=npatient
        avg_ssim_PVS/=npatient
        avg_psnr_WM/=npatient
        avg_ssim_WM/=npatient
        
        name.append("average")
        PVS_psnr_list.append(avg_psnr_PVS)
        PVS_ssim_list.append(avg_ssim_PVS)
        WM_psnr_list.append(avg_psnr_WM)
        WM_ssim_list.append(avg_ssim_WM)
    
        print("avg_psnr_PVS = {:.5f}".format(avg_psnr_PVS))
        print("avg_ssim_PVS = {:.5f}".format(avg_ssim_PVS))
        print("avg_psnr_WM = {:.5f}".format(avg_psnr_WM))
        print("avg_ssim_WM = {:.5f}".format(avg_ssim_WM))
        print(" ")
        
        df = pd.DataFrame({"patient":name,"PVS_psnr": PVS_psnr_list,"PVS_ssim":PVS_ssim_list,"WM_psnr":WM_psnr_list,"WM_ssim":WM_ssim_list})
            
        writer = pd.ExcelWriter(testout_path+"/"+network_name+load_modelpath[-3:]+".xlsx", engine='xlsxwriter')
        
        df.to_excel(writer, sheet_name='Sheet1')
        
        writer.save()
        
        print("save excel done")
        
    


if __name__ == '__main__':
  tf.app.run()

