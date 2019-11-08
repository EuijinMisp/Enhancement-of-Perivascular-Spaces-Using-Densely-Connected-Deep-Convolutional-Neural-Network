# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:10:46 2018

@author: Owner
"""

import os
import SimpleITK as sitk
import numpy as np
import math
import random
from sklearn.utils import shuffle

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img-mean)/std
    img[img>10]=10
    img[img<-1]=-1
    img = (img+1)/11 
    return img

def img_crop(img,center,size):
    z=center[0]
    y=center[1]
    x=center[2]
    
    dz = size[0]
    dy = size[1]
    dx = size[2]
    
    if dz == 0:
        img = img[z,y-int(dy/2):y+int(dy/2),x-int(dx/2):x+int(dx/2)]
    elif dy == 0:
        img = img[z-int(dz/2):z+int(dz/2),y,x-int(dx/2):x+int(dx/2)]
    elif dx == 0:
        img = img[z-int(dz/2):z+int(dz/2),y-int(dy/2):y+int(dy/2),x]
    
    img = img*254.9

    return img

def medi_imread(path,file_name):
    
    img_file = sitk.ReadImage(os.path.join(path,file_name)) # img load
    img = sitk.GetArrayFromImage(img_file)
    
    return img,img_file


def ssim3d(img1, img2):
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)
    sig_x = np.std(img1)
    sig_y = np.std(img2)
    sig_xy = np.mean((img1 - mu_x)*(img2-mu_y))
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    C3 = C2/2

    ssim = ((2*mu_x*mu_y+C1)*(2*sig_xy+C2)*(sig_xy+C3))/(((mu_x*mu_x)+(mu_y*mu_y)+C1)*((sig_x*sig_x)+(sig_y*sig_y)+C2)*(sig_x*sig_y+C3))

    return ssim


def save_3dimg (save_path,file_name,img_file,img):

    seg2_file = sitk.GetImageFromArray(img)
    spacing = img_file.GetSpacing()
    seg2_file.SetSpacing(spacing)
    origin = img_file.GetOrigin()
    seg2_file.SetOrigin(origin)
    direction = img_file.GetDirection()
    seg2_file.SetDirection(direction)
    sitk.WriteImage(seg2_file, os.path.join(save_path,file_name))
    
    return "image saved"


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Medical_data:
    
    def __init__(self,img_path,label_path,seg_path,pvs_seg_path,img_patch_st,img_patch_size):
        
        self.img_path = img_path
        
        self.label_path = label_path
        
        self.seg_path =seg_path
        
        self.pvs_seg_path = pvs_seg_path

        self.img_patch_st = img_patch_st

        self.stride = np.copy(img_patch_st)
              
        self.img_patch_size = img_patch_size
        
        self.gap = self.img_patch_size-self.img_patch_st
        
        self.h_gap = int(self.gap/2)

                   
    def dataload(self,n_patient,data_type,mode,val_order=1):
                                  
        patients = os.listdir(self.img_path)
        patients_list=[]

        for i in range(len(patients)):
            if patients[i][-len(data_type):]==data_type:
                patients_list.append(patients[i])
        
        img_list = []
        label_list = []
        seg_list=[]
        WM_list=[]
        PVS_list=[]
        brain_label=[]

        self.shape_list=[]
 
        #mean_var_list=[]
        self.depcut = {"229.img":20,
                  "230.img":20,
                  "232.img":20,
                  "236.img":40,
                  "237.img":40,
                  "238.img":30,
                  "239.img":30,
                  "240.img":30,
                  "241.img":30,
                  "1203.img":30,
                  "242.img":10,
                  "243.img":20,
                  "244.img":20,
                  "245.img":30,
                  "250.img":20,
                  "251.img":20,
                  "261.img":10}

        if mode=="train":        
            self.img_patch_st = random.randrange(self.stride-5,self.stride+1)


        for patient in patients_list[:n_patient]:

            medical_img,medical_file = medi_imread(self.img_path,patient) 
            medical_label,medical_file = medi_imread(self.label_path,patient) 
            seg_img,seg_file = medi_imread(self.seg_path,patient)
            pvs_seg,pvs_file = medi_imread(self.pvs_seg_path,patient)

            medical_img = normalize(medical_img)
            medical_label = normalize(medical_label)
            
            medical_img = medical_img[self.depcut[patient]:,:,:]##crop edge : remove artifact of original image          
            medical_label = medical_label[self.depcut[patient]:,:,:]             
            seg_img = seg_img[self.depcut[patient]:,:,:]
            pvs_seg = pvs_seg[self.depcut[patient]:,:,:]
            
            WM_mask=np.copy(seg_img)
            WM_mask[WM_mask!=3]=0
            WM_mask[WM_mask==3]=1

            wm_img = medical_img * WM_mask          
            wm_label = medical_label * WM_mask          
            self.shape_list.append(medical_img.shape)            
            
            seg_img_crop = seg_img[self.h_gap:seg_img.shape[0]-self.h_gap,self.h_gap:seg_img.shape[1]-self.h_gap,self.h_gap:seg_img.shape[2]-self.h_gap]
            seg_list.append(seg_img_crop)
            
            pvs_seg_crop = pvs_seg[self.h_gap:pvs_seg.shape[0]-self.h_gap,self.h_gap:pvs_seg.shape[1]-self.h_gap,self.h_gap:pvs_seg.shape[2]-self.h_gap]
            PVS_list.append(pvs_seg_crop)                          

            WM_mask_crop = WM_mask[self.h_gap:WM_mask.shape[0]-self.h_gap,self.h_gap:WM_mask.shape[1]-self.h_gap,self.h_gap:WM_mask.shape[2]-self.h_gap]           
            WM_list.append(WM_mask_crop)

            medical_label = medical_label[self.h_gap:WM_mask.shape[0]-self.h_gap,self.h_gap:WM_mask.shape[1]-self.h_gap,self.h_gap:WM_mask.shape[2]-self.h_gap]           
            brain_label.append(medical_label)  
            
            nnz = medical_img.shape[0]
            nny = medical_img.shape[1]
            nnx = medical_img.shape[2]
            
                          
            for i in range(0,nnz,self.img_patch_st):
                for j in range(0,nny,self.img_patch_st):
                    for k in range(0,nnx,self.img_patch_st):
                        
                        if (i+self.img_patch_size <= nnz) and (j+self.img_patch_size <= nny) and (k+self.img_patch_size <= nnx):
                            
                            patch_img = wm_img[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            patch_label = wm_label[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            patch_bin = WM_mask[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            
                            if mode == "train":                                    
                                if np.sum(patch_bin)>4000 :
                                    img_list.append(patch_img)
                                    label_list.append(patch_label)
                                                                         
                            else:                               
                                img_list.append(patch_img)
                                label_list.append(patch_label)
                
        
        img_list = np.array(img_list)
        label_list = np.array(label_list)

        img_list = img_list[:,:,:,:,np.newaxis]
        label_list = label_list[:,:,:,:,np.newaxis]        

        if mode=="train":
            img_list, label_list = shuffle(img_list, label_list)
            img_list = img_list[:2000]
            label_list = label_list[:2000]
            return img_list, label_list

        elif mode=="test":
            return img_list, label_list,seg_list,WM_list,PVS_list,brain_label

    def recon_3d(self,test_out,n_patient):
        
        recon_list=[]
        cnt=0

        for i in range(n_patient):        
            
            z = self.shape_list[i][0] - self.gap
            y = self.shape_list[i][1] - self.gap        
            x = self.shape_list[i][2] - self.gap    
            
            recon_out = np.zeros((z,y,x))

            stich = self.img_patch_st
            
            step0 = int(z/stich)
            
            step1 = int(y/stich)
            
            step2 = int(x/stich)
            
            for i in range(step0):
                for j in range(step1):
                    for k in range(step2):
                        
                        recon_out[stich*i:stich*i+stich, stich*j:stich*j+stich, stich*k:stich*k+stich] = test_out[cnt]
                        cnt+=1
 
            recon_list.append(recon_out)
        
        return recon_list
    
    

 



        

    

