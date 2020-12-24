'''
In this folder there are many pickle files that map to the data
/data/bigbone6/fcaliva/VBR_python/pickle_files/

to launch atom from terminal:
/netopt/rhel7/versions/atom/latest/atom


You can use my environment by using these 3 lines the first time:
/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda init tcsh
cp /data/knee_mri8/Francesco/.condarc ~/
conda activate pytorch_env
conda activate tf_env
from the second time
conda activate pytorch_env
conda activate tf_env

in general they are in /data/knee_mri8/Francesco/env
'''
import pickle
import pydicom
from scipy.io import loadmat
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
def load_pickle(path):
    return pickle.load(open(path,'rb'))
def load_dicom(path):
    return pydicom.read_file(path)
def load_mat(path):
    return loadmat(path)
def save_pickle(var,path):
    pickle.dump(var,open(path,'wb'))

def find_parameters_physical_alignment_to_oai_size(mri_info):
    nr_atlas,nc_atlas = 384,384
    row_pix_spacing_atlas, col_pix_spacing_atlas = 0.3125, 0.3125
    fov_row_atlas, fov_col_atlas = nr_atlas*row_pix_spacing_atlas, nc_atlas*col_pix_spacing_atlas

    nr_cartigram, nc_cartigram = mri_info.Rows,mri_info.Columns
    row_pix_spacing_cartigram, col_pix_spacing_cartigram = mri_info.PixelSpacing
    fov_row_cartigram, fov_col_cartigram = nr_cartigram*row_pix_spacing_cartigram, nc_cartigram*col_pix_spacing_cartigram

    desired_acquisition_matrix_row,desired_acquisition_matrix_col = fov_row_atlas//row_pix_spacing_cartigram,fov_col_atlas/col_pix_spacing_cartigram
    rem_row, rem_col = int(nr_cartigram - desired_acquisition_matrix_row)//2, int(nc_cartigram - desired_acquisition_matrix_col)//2
    return (nr_atlas,nc_atlas,rem_row, rem_col)

def convert_to_oai_size(volume, tuple_parameter):
    nr_atlas,nc_atlas, rem_row, rem_col= tuple_parameter
    if rem_row != 0:
        resized = volume[rem_row:-rem_row,...]
    else:
        resized = volume[rem_row:,...]

    if rem_col != 0:
        resized = resized[:,rem_col:-rem_col]
    else:
        resized = resized[:,rem_col:]

    new_resized= np.array(Image.fromarray(resized.astype(np.float32)).resize((nr_atlas,nc_atlas)))

    return new_resized.astype(volume.dtype)

def load_oai(x,nclasses = 6):
    ['Background','LTC','LFC','MTC','MFC','PC']
    mri_info = load_dicom(x[0])
    mri=(mri_info.pixel_array/x[-1])[20:-20,20:-20,np.newaxis]
    seg_temp= load_mat(x[1])['bin_mask'][20:-20,20:-20,x[2]]
    seg = np.zeros((seg_temp.shape[0], seg_temp.shape[1],nclasses)).astype('uint8')
    for idclasses in range(nclasses):
        seg[:,:,idclasses] = (seg_temp==idclasses).astype('uint8')
    # [[plt.imshow(seg[...,cl]),plt.show()] for cl in range(6)]
    return mri, seg

def load_ucsf(x='',nclasses = 6):
    mri_info = load_dicom(x[0])
    mri = convert_to_oai_size(volume=mri_info.pixel_array, tuple_parameter=find_parameters_physical_alignment_to_oai_size(mri_info)).astype(np.float32)
    mri=mri[20:-20,20:-20,np.newaxis]
    all_max_values_ucsf_volumes=load_pickle('/data/bigbone6/fcaliva/VBR_python/max_values_ucsf_volumes.pickle')
    max_value = all_max_values_ucsf_volumes[x[0].split('/')[-2]]['et'+str(np.array(mri_info.EchoTime,dtype=np.float32))]

    mri/=max_value
    nr,nc = mri.shape[:-1]
    # need to create the background
    seg = np.zeros((nr,nc,nclasses))
    for cl in range(nclasses-1):
        seg_info = load_dicom(x[1][cl])
        seg_temp = convert_to_oai_size(volume=seg_info.pixel_array, tuple_parameter=find_parameters_physical_alignment_to_oai_size(seg_info)).astype(np.float32)
        seg[:,:,cl+1] = seg_temp[20:-20,20:-20]
    # create the background
    bkg = 1- seg[...,1:].sum(-1)
    bkg[bkg>1]=0
    bkg[bkg<0]=0
    seg[:,:,0] = bkg.astype('uint8')
    # [[plt.imshow(seg[...,cl]),plt.show()] for cl in range(6)]
    return mri, seg

split = 'train'
for split in ['train','valid','text']:
    # balanced = same number of files from OAI and UCSF
  pickle_name = f'/data/bigbone6/fcaliva/VBR_python/pickle_files/segmented_slices_oai_and_ucsf_fixed_{split}.pickle'
  # not balanced = all OAI and UCSF
  # pickle_name = f'/data/bigbone6/fcaliva/VBR_python/pickle_files/segmented_slices_oai_and_ucsf_fixed_ALLfiles_{split}.pickle'

  pickle_cont = load_pickle(pickle_name)

  for x in pickle_cont:
    if '.mat' in x[1]:
        img,seg = load_oai(x)
    else:
        img,seg = load_ucsf(x)
# plot.imshow(img[...,0])
# plot.imshow(seg[...,3])

'''
volumes of Echo1 of oai files, in mat format are stored in these pickles
'''
oai_train = load_pickle('/data/knee_mri4/segchal_fra/Dioscorides/splits/t2cartilagesegmentation/split_train.pickle')
im = load_mat(oai_train[0][0])['true_img']
im.shape
seg = load_mat(oai_train[0][1])['bin_mask']


'''
to create a volume for the cartigram, you can go to '/data/bigbone6/fcaliva/VBR_python/file_for_Sarthak.py'
a complete exame on how to use it from import image to t2 map calculation /data/bigbone6/fcaliva/VBR_python/main.py
following is an example
'''

cd /data/bigbone6/fcaliva/VBR_python
import os
import pickle
import yaml
import argparse
from cartigram_VBR import *
import logging

import inspect
obj = Main('','/data/bigbone6/fcaliva/VBR_python/acquisitions/original/UCSF/echoes/CART06')
obj.cartigrams[0].
inspect.getmembers(obj.cartigrams[0])
