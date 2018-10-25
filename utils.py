# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:10:46 2018

@author: Owner
"""

import os
import SimpleITK as sitk
import numpy as np
import math
import cv2
import pandas as pd
import re
from scipy import signal
#from skimage import morphology
from sklearn.utils import shuffle
import random

from sklearn.utils import shuffle

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dilation(img):
    
    img = morphology.dilation(img,kernel)

    return img

def erosion(img):
    img = morphology.erosion(img,kernel)    
    return img
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
    
    return img

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

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

"""
def ssim(img1, img2, cs_map=False):
    

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 
        (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return np.mean(value)   

def ssim3d(img1, img2, cs_map=False):
    avg_ssim=0
    for i in range(img1.shape[0]):
        avg_ssim += ssim(img1[i], img2[i])
    
    avg_ssim/=  img1.shape[0]        
    
    return avg_ssim
"""

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

def write_excel(x_data,y_data,x_axis,y_axis,save_path):
    
    df = pd.DataFrame({y_axis:y_data,x_axis: x_data})
    
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    
    df.to_excel(writer, sheet_name='Sheet1')
    
    writer.save()
    
    return print("save excel done")


class Medical_data:
    
    def __init__(self,img_path,label_path,seg_path,pvs_seg_path,img_patch_st,\
                 img_patch_size):
        
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
        patients_label = os.listdir(self.label_path)
        patients_list=[]
        patients_label_list=[]
        
        for i in range(len(patients)):
            if patients[i][-len(data_type):]==data_type:
                patients_list.append(patients[i])
        for i in range(len(patients_label)):
            if patients_label[i][-len(data_type):]==data_type:
                patients_label_list.append(patients_label[i])            
                     
        patients.sort()        
        patients_label.sort()
        
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
        
        if mode=="validation":
            patient = patients_list[val_order]
            patient_label = patients_label_list[val_order]
            
            patients_list=[]
            patients_label_list=[]
            
            patients_list.append(patient)
            patients_label_list.append(patient_label)
        if mode=="train":        
            self.img_patch_st = random.randrange(self.stride-5,self.stride+1)
            #print("stride")
            #print(self.img_patch_st)
            #print()

        for patient in patients_list[:n_patient]:
            
            medical_file = sitk.ReadImage(os.path.join(self.img_path,patient)) # img load
            medical_img = sitk.GetArrayFromImage(medical_file)
            medical_img = normalize(medical_img)

            seg_file = sitk.ReadImage(os.path.join(self.seg_path,patient)) # img load
            seg_img = sitk.GetArrayFromImage(seg_file)
            
            pvs_file = sitk.ReadImage(os.path.join(self.pvs_seg_path,patient)) # img load
            pvs_seg = sitk.GetArrayFromImage(pvs_file)

            medical_img = medical_img[self.depcut[patient]:,:,:]##crop edge                
            seg_img = seg_img[self.depcut[patient]:,:,:]##crop edge
            pvs_seg = pvs_seg[self.depcut[patient]:,:,:]##crop edge
            
            WM_mask=np.copy(seg_img)
            WM_mask[WM_mask!=3]=0
            WM_mask[WM_mask==3]=1

            wm_img = medical_img * WM_mask
            
            self.shape_list.append(medical_img.shape)
            
            
            seg_img_crop = seg_img[self.h_gap:seg_img.shape[0]-self.h_gap,self.h_gap:seg_img.shape[1]-self.h_gap,self.h_gap:seg_img.shape[2]-self.h_gap]
            seg_list.append(seg_img_crop)
            
            pvs_seg_crop = pvs_seg[self.h_gap:pvs_seg.shape[0]-self.h_gap,self.h_gap:pvs_seg.shape[1]-self.h_gap,self.h_gap:pvs_seg.shape[2]-self.h_gap]
            PVS_list.append(pvs_seg_crop)                          

            WM_mask_crop = WM_mask[self.h_gap:WM_mask.shape[0]-self.h_gap,self.h_gap:WM_mask.shape[1]-self.h_gap,self.h_gap:WM_mask.shape[2]-self.h_gap]           
            WM_list.append(WM_mask_crop)
            
            nnz = medical_img.shape[0]
            nny = medical_img.shape[1]
            nnx = medical_img.shape[2]
            
                          
            for i in range(0,nnz,self.img_patch_st):
                for j in range(0,nny,self.img_patch_st):
                    for k in range(0,nnx,self.img_patch_st):
                        
                        if (i+self.img_patch_size <= nnz) and (j+self.img_patch_size <= nny) and (k+self.img_patch_size <= nnx):
                            
                            patch = wm_img[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            patch2 = WM_mask[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            
                            if mode == "train":                                    
                                if np.sum(patch2)>4000 :
                                    img_list.append(patch)                                     
                            else:                               
                                img_list.append(patch)
        

        for patient in patients_label_list[:n_patient]:
            
            label_file = sitk.ReadImage(os.path.join(self.label_path, patient ))
            label_img = sitk.GetArrayFromImage(label_file)# label data load   
            label_img = normalize(label_img)
            
            seg_file = sitk.ReadImage(os.path.join(self.seg_path,patient)) # img load
            seg_img = sitk.GetArrayFromImage(seg_file)
            
            label_img = label_img[self.depcut[patient]:,:,:] ##crop edge
            seg_img = seg_img[self.depcut[patient]:,:,:]##crop edge
            
            WM_mask = np.copy(seg_img)
            WM_mask[WM_mask!=3]=0
            WM_mask[WM_mask==3]=1
            seg_img[seg_img>0]=1

            label_wm = label_img*WM_mask
            label_brain = label_img*seg_img
            label_brain = label_brain[self.h_gap:seg_img.shape[0]-self.h_gap,self.h_gap:seg_img.shape[1]-self.h_gap,self.h_gap:seg_img.shape[2]-self.h_gap]
            brain_label.append(label_brain)      

            nnz = label_img.shape[0]              
            nny = label_img.shape[1]
            nnx = label_img.shape[2]    
                        
            for i in range(0,nnz,self.img_patch_st):
                for j in range(0,nny,self.img_patch_st):
                    for k in range(0,nnx,self.img_patch_st):
                        
                        if (i+self.img_patch_size <= nnz) and (j+self.img_patch_size <= nny) and (k+self.img_patch_size <= nnx):
                            
                            patch = label_wm[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            patch2 = WM_mask[i:i+self.img_patch_size,j:j+self.img_patch_size,k:k+self.img_patch_size]
                            
                            if mode == "train":
                                
                                if np.sum(patch2)>4000 :
                                    label_list.append(patch)
                           
                            else:                               
                                label_list.append(patch)
                   
        
        img_list = np.array(img_list)
        label_list = np.array(label_list)
     
        img = np.zeros((len(img_list),self.img_patch_size,self.img_patch_size,self.img_patch_size,1))
        img[:,:,:,:,0] = img_list[:,:,:,:]
        
        label = np.zeros((len(label_list),self.img_patch_size,self.img_patch_size,self.img_patch_size,1))
        label[:,:,:,:,0] = label_list[:,:,:,:]        
        
        if mode=="train":
            img, label = shuffle(img, label)
            img = img[:2000]
            label = label[:2000]
            #print(img.shape)
            return img, label
        elif mode=="test":
            return img, label,seg_list,WM_list,PVS_list,brain_label
        elif mode=="validation":
            return img, label,WM_mask_crop,pvs_seg_crop

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
    
    

 



        

    

