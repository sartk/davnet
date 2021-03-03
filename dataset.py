import pickle
import pydicom
from scipy.io import loadmat
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
import torch
from torch.utils.data import Dataset

class kMRI(Dataset):

    def __init__(self, src, balanced=True, pickle_name=None, group='all'):

        assert group in ['all', 'source', 'target'], "group must be one of: 'all', 'source', 'target'"
        assert src in ['train', 'test', 'valid'], "src must be one of: 'train', 'test', 'valid'"

        if pickle_name is None:
            if balanced:
                pickle_name = f'/data/bigbone6/fcaliva/VBR_python/new_pickle_files/ucsf_and_oai_only_segmented_slices_{src}.pickle'
            else:
                pickle_name = f'/data/bigbone6/fcaliva/VBR_python/new_pickle_files/all_files_oai_and_ucsf_only_segmented_slices_{src}.pickle'

        if group == 'all':
            h = lambda x: True
        elif group == 'source':
            h = lambda x: '.mat' in x[1]
        elif group == 'target':
            h = lambda x: '.mat' not in x[1]

        self.data = list(filter(h, load_pickle(pickle_name)))
        self.src = src

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, meta=False, dangerous=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        meta = self.data[idx]
        if dangerous:
            meta = ['/working/oaidl1/00m/0.C.3/9574138/20050209/10223410/022', '/data/knee_mri4/alaleh_all/vnet_T2_data/T2_val_mask_cleaned/9574138_V00.mat', 19, 1, 1144, 1396]
        if '.mat' in meta[1]:
            img, seg = load_oai(meta)
            domain = torch.tensor([1, 0])
        else:
            img, seg = load_ucsf(meta)
            domain = torch.tensor([0, 1])
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        seg = torch.from_numpy(seg).permute(2, 0, 1).contiguous()

        return img.float(), seg.float(), domain.long(), meta


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

def load_oai(x,nclasses = 4):
    num_all_classes = 6
    mri_info = load_dicom(x[0])
    mri=(mri_info.pixel_array/x[-1])[20:-20,20:-20,np.newaxis]
    mri/=np.percentile(mri,85)
    seg_temp= load_mat(x[1])['bin_mask'][20:-20,20:-20,x[2]]
    nr,nc = seg_temp.shape
    seg = np.zeros((nr, nc,num_all_classes)).astype('uint8')
    for idclasses in range(num_all_classes):
        # {0: 'background', 1:'LTC',2:'LFC',3:'MTC',4:'MFC',5:'PC'}
        seg[...,idclasses] = (seg_temp==idclasses).astype('uint8')
    # [[plt.imshow(seg[...,cl]),plt.show()] for cl in range(6)]
    if nclasses == 4:
        # {0: 'background', 1:'TC',2:'FC',3: 'PC'}
        seg_final = np.zeros((nr, nc, nclasses))
        seg_final[...,0] = seg[...,0]
        seg_final[...,1] = seg[...,1] + seg[...,3]
        seg_final[...,2] = seg[...,2] + seg[...,4]
        seg_final[...,3] = seg[...,5]
        return mri, seg_final
    return mri, seg

def load_ucsf(x='',nclasses =4):
    mri_info = load_dicom(x[0])
    num_all_classes = 6
    nr, nc = 344, 344
    mri = convert_to_oai_size(volume=mri_info.pixel_array, tuple_parameter=find_parameters_physical_alignment_to_oai_size(mri_info)).astype(np.float32)
    mri=mri[20:-20,20:-20,np.newaxis]
    all_max_values_ucsf_volumes=load_pickle('/data/bigbone6/fcaliva/VBR_python/max_values_ucsf_volumes.pickle')
    max_value = all_max_values_ucsf_volumes[x[0].split('/')[-2]]['et'+str(np.array(mri_info.EchoTime,dtype=np.float32))]

    mri/=max_value
    mri/=np.percentile(mri,85)
    nr, nc = mri.shape[:-1]
    # need to create the background
    seg = np.zeros((nr, nc, num_all_classes))
    for cl in range(num_all_classes - 1):
        seg_info = load_dicom(x[1][cl])
        seg_temp = convert_to_oai_size(volume=seg_info.pixel_array, tuple_parameter=find_parameters_physical_alignment_to_oai_size(seg_info)).astype(np.float32)
        seg[:, :, cl + 1] = seg_temp[20:-20, 20:-20]
    # create the background
    bkg = 1- seg[...,1:].sum(-1)
    bkg[bkg>1]=0
    bkg[bkg<0]=0
    seg[:,:,0] = bkg.astype('uint8')
    # [[plt.imshow(seg[...,cl]),plt.show()] for cl in range(6)]
    if nclasses == 4:
        # {0: 'background', 1:'TC',2:'FC',3: 'PC'}
        seg_final = np.zeros((nr, nc, nclasses))
        seg_final[...,0] = seg[...,0]
        seg_final[...,1] = seg[...,1] + seg[...,3]
        seg_final[...,2] = seg[...,2] + seg[...,4]
        seg_final[...,3] = seg[...,5]
        return mri, seg_final
    return mri, seg
