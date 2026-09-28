import io
import os
import time
import pydicom as dicom
import torch.utils.data as data
from PIL import Image
import numpy as np
import pdb
import torch
import random
import pickle
import matplotlib.pyplot as plt
from typing import Any, Callable
from math import cos, sin
import torchvision
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


def findCTDepth(ct):
    ctOrigX, ctOrigY, ctOrigZ = ct.ImagePositionPatient
    return ctOrigZ+0

def readWholeCTFunction(ctdcms, start_x, start_y, start_z, end_x, end_y, end_z, dose_shape):
    ctSizeZ = len(ctdcms)
    ctdcms = sorted(ctdcms, key=findCTDepth)
     

    c1 = ctdcms[0]
    c2 = ctdcms[1]
    ctstart_x, ctstart_y, ctstart_z = c1.ImagePositionPatient
    ctSpacZ = c2.ImagePositionPatient[2] - ctstart_z
    ctSpacX, ctSpacY = c1.PixelSpacing


    ctstart_zindices = int(np.floor((start_z - ctstart_z) /ctSpacZ))
    ctend_zindices = int(np.ceil((end_z - start_z) /ctSpacZ)) +ctstart_zindices

    ctstart_yindices = int(np.floor((start_y - ctstart_y) /ctSpacY))
    ctend_yindices = int(np.ceil((end_y - start_y) /ctSpacY)) +ctstart_yindices

    ctstart_xindices = int(np.floor((start_x - ctstart_x) /ctSpacX))
    ctend_xindices = int(np.ceil((end_x - start_x) /ctSpacX)) +ctstart_xindices

    RescaleIntercept = float(c1.RescaleIntercept)
    RescaleSlope = float(c1.RescaleSlope)
    ct = np.zeros((ctend_zindices -ctstart_zindices+1, ctend_yindices -ctstart_yindices+1, ctend_xindices -ctstart_xindices+1),dtype = np.float32)
    if ctstart_zindices ==-1:
        ctend_zindices = ctend_zindices+1
        ctstart_zindices = 0
    if ctstart_yindices==-1:
        ctend_yindices = ctend_yindices+1
        ctstart_yindices = 0 
    if ctstart_xindices ==-1 :
        ctend_xindices = ctend_xindices+1
        ctstart_zindices = 0
    for i in range(ct.shape[0]):
        ctdcm = ctdcms[i+ctstart_zindices]
        tmp = np.array(ctdcm.pixel_array, dtype = np.float32)*RescaleSlope + RescaleIntercept
        tmp  = tmp[ctstart_yindices: ctend_yindices+1, ctstart_xindices:ctend_xindices+1]
        ct[i,:,:] = tmp
     
    zoom_factors = (dose_shape[0]/ct.shape[0], dose_shape[1]/ct.shape[1], dose_shape[2]/ct.shape[2]) #(z, y, x)
    ct = zoom(ct, zoom_factors, order=3)
     
    ct[ct<-1000] = -1000
    mapped_array = np.zeros_like(ct, dtype=np.float32)

    # Map -1000 to -200 -> 0 to 0.1
    mask1 = ct < -200
    mapped_array[mask1] = 0 + (ct[mask1] - (-1000)) / (-200 - (-1000)) * 0.1
    
    # Map -200 to 300 -> 0.1 to 0.7
    mask2 = (-200 <= ct) & (ct < 300)
    mapped_array[mask2] = 0.1 + (ct[mask2] - (-200)) / (300 - (-200)) * (0.7 - 0.1)
    
    # Map 300 to 3000 -> 0.7 to 0.85
    mask3 = (300 <= ct) & (ct < 3000)
    mapped_array[mask3] = 0.7 + (ct[mask3] - 300) / (3000 - 300) * (0.85 - 0.7)
    
    # Map 3000 to 29000 -> 0.85 to 1 (nonlinear)
    mask4 = (3000 <= ct)
    mapped_array[mask4] = 0.85 + np.log(ct[mask4] - 3000 + 1) / np.log(29000 - 3000 + 1) * (1 - 0.85)
    ct = mapped_array
    return ct


def make_dataset(dir):
    samples = []
    test_samples = []

    if not os.path.isfile(os.getcwd()+'/data_list_pkl/dose/sample_list_'+dir.split('/')[-1]+'4x32x32_Gy_1pfiltered.pkl'): #### constructing the dataset
        dir = os.getcwd().split('DiT')[0]+'dose_data'
        
        for pat in os.listdir(dir):
            ctdcms = []
            file_folder = os.path.join(dir,pat)
            dcmfiles = os.listdir(file_folder)          
            for dcmfile in dcmfiles:
                dcmfileDir = os.path.join(file_folder, dcmfile) 
                dcmType = dcmfile.split('.')[0]
                if dcmType == 'ct':
                    dcm = dicom.dcmread(dcmfileDir)
                    ctdcms.append(dcm)
                else:
                    pass
            # find the starting points and ending points from file.
            paired_fullname = np.load(os.path.join(file_folder,'paired_fullname_cliptoBODY.npy'))
             
            for idx in range(len(paired_fullname)):          
                Hdose_dicom = dicom.dcmread(os.path.join(file_folder,paired_fullname[idx][0]))
                Ldose_dicom = dicom.dcmread(os.path.join(file_folder,paired_fullname[idx][1]))
                Hdose_shape = Hdose_dicom.pixel_array.shape
                Hdose_spacing_z = Hdose_dicom.GridFrameOffsetVector[1] - Hdose_dicom.GridFrameOffsetVector[0]
                Hdose_spacing_x, Hdose_spacing_y = Hdose_dicom.PixelSpacing[0], Hdose_dicom.PixelSpacing[1]
                start_x, start_y, start_z = float(Hdose_dicom.ImagePositionPatient[0]), float(Hdose_dicom.ImagePositionPatient[1]), float(Hdose_dicom.ImagePositionPatient[2])

                end_x = start_x + (Hdose_shape[-1] - 1) * Hdose_spacing_x
                end_y = start_y + (Hdose_shape[1] - 1) * Hdose_spacing_y
                end_z = start_z + (Hdose_shape[0] - 1) * Hdose_spacing_z
                POICT = readWholeCTFunction(ctdcms, start_x, start_y, start_z, end_x, end_y, end_z, Hdose_shape)
                POICT = POICT.reshape(-1)

                ldose_imaging = np.array(Ldose_dicom.pixel_array)
                hdose_imaging = np.array(Hdose_dicom.pixel_array)
                ldose_imaging[ldose_imaging<1000] = 0
                hdose_imaging[hdose_imaging<10] = 0

                ldose_imaging = ldose_imaging*Ldose_dicom.DoseGridScaling
                hdose_imaging = hdose_imaging*Hdose_dicom.DoseGridScaling
                ldose_imagingG = ldose_imaging.max()*gaussian_filter(ldose_imaging/ldose_imaging.max(), sigma=1)  

                shapestr = str(ldose_imaging.shape[0])+'_'+str(ldose_imaging.shape[1])+'_'+str(ldose_imaging.shape[2])
                 
                ldose_imaging = ldose_imaging.reshape(-1, order='F')
                hdose_imaging = hdose_imaging.reshape(-1, order='F')
                ldose_imagingG = ldose_imagingG.reshape(-1, order='F')

                if POICT.shape[0] <= 4*32*32:
                    if not (ldose_imaging.sum()==0 or hdose_imaging.sum()==0):
                        tmpCT = np.concatenate((POICT, np.zeros(4*32*32-POICT.shape[0])))
                        tmpldose_imaging = np.concatenate((ldose_imagingG, np.zeros(4*32*32-POICT.shape[0])))
                        tmphdose_imaging = np.concatenate((hdose_imaging, np.zeros(4*32*32-POICT.shape[0])))
                        prefix = 'G_Forder_Gy_'+pat+'_size'+shapestr+'_chunk0_'       
                        np.save('../dose4x32x32/'+prefix+'all.npy',np.stack((tmpCT.reshape((4,32,32)),tmpldose_imaging.reshape((4,32,32)),tmphdose_imaging.reshape((4,32,32)))))
                        item = (prefix)
                        if pat in test_pat:
                            test_samples.append(item)
                        else:
                            samples.append(item)
                else:
                    NumofPatch = POICT.shape[0]//(4*32*32)
                    for i in range(NumofPatch):
                        if not (ldose_imaging[i*4*32*32:(i+1)*4*32*32].sum()==0 or hdose_imaging[i*4*32*32:(i+1)*4*32*32].sum()==0):
                            tmpCT = POICT[i*4*32*32:(i+1)*4*32*32]
                            tmpldose_imaging = ldose_imagingG[i*4*32*32:(i+1)*4*32*32]
                            tmphdose_imaging = hdose_imaging[i*4*32*32:(i+1)*4*32*32]
                            prefix = 'G_Forder_Gy_'+pat+'_size'+shapestr+'_chunk'+str(i)+'_'
                            np.save('../dose4x32x32/'+prefix+'all.npy',np.stack((tmpCT.reshape((4,32,32)),tmpldose_imaging.reshape((4,32,32)),tmphdose_imaging.reshape((4,32,32)))))
                            item = (prefix)
                            if pat in test_pat:
                                test_samples.append(item)
                            else:
                                samples.append(item)
                    if not (ldose_imaging[(i+1)*4*32*32:].sum()==0 or hdose_imaging[(i+1)*4*32*32:].sum()==0):
                        tmpCT = np.concatenate((POICT[(i+1)*4*32*32:], np.zeros((i+2)*4*32*32-POICT.shape[0])))
                        tmpldose_imaging = np.concatenate((ldose_imagingG[(i+1)*4*32*32:], np.zeros((i+2)*4*32*32-POICT.shape[0])))
                        tmphdose_imaging = np.concatenate((hdose_imaging[(i+1)*4*32*32:], np.zeros((i+2)*4*32*32-POICT.shape[0])))
                        prefix = 'G_Forder_Gy_'+pat+'_size'+shapestr+'_chunk'+str(i+1)+'_'
                        np.save('../dose4x32x32/'+prefix+'all.npy',np.stack((tmpCT.reshape((4,32,32)),tmpldose_imaging.reshape((4,32,32)),tmphdose_imaging.reshape((4,32,32)))))
                        item = (prefix)
                        if pat in test_pat:
                            test_samples.append(item)
                        else:
                            samples.append(item)

        
        pickle.dump(samples, open('./data_list_pkl/dose/sample_list_train4x32x32_Gy_Forder.pkl', 'wb'))
        pickle.dump(test_samples, open('./data_list_pkl/dose/sample_list_test4x32x32_Gy_Forder.pkl', 'wb'))


        return samples

    
        
    else:

        samples = np.load('./data_list_pkl/dose/sample_list_train4x32x32_Gy_Forder.pkl',allow_pickle=True)
        print(len(samples))
        return samples





def default_loader(path: str) -> Any:
    return np.load(path)


class DatasetFolder(data.Dataset):
    def __init__(self, root, img_prefix='', loader: Callable[[str], Any] = default_loader, transform=None, cache_mode="no"):                                                                                                                                                                            
                 
        # image folder mode

         
        self.root = root
        self.transform = transform
        self.loader = loader        


        self.samples = make_dataset(os.path.join(root, img_prefix))     



        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (IM_low, IM_high) where IM_high is high-N IM of corresponding low-N IM.
        """
         
        if 'raw' in self.root:
            self.root = self.root.split('_')[0]
        path = os.path.join(self.root, self.samples[index])        
        CT, IM_low, IM_high = self.loader(path+'CT.pkl.npy'), self.loader(path+'iml.npy'), self.loader(path+'imh.npy')
        if self.transform is not None:
            IM_low = self.transform(IM_low)
            IM_high = self.transform(IM_high)

        return IM_low, IM_high

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class IM_ImageFolder (DatasetFolder):
    def __init__(self, root, img_prefix='', loader: Callable[[str], Any] = default_loader, transform=None):
        super(IM_ImageFolder, self).__init__(root,img_prefix=img_prefix, loader=default_loader, transform = transform)
        self.img_prefix = img_prefix
        self.root = root
        self.transform = transform
        self.loader = loader
        self.prot = 0.75
        self.pflip = 2/3
        


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, IM_high).
        """
        
         
        if 'raw' in self.root:
            self.root = self.root.split('_')[0]
        path = os.path.join(self.root, self.samples[index]) 
        img  = self.loader(path+'all.npy')
         


        

        CT, IM_low, IM_high = img[0,:], img[1,:],img[2,:]
        CT[CT<0] = 0

        # mask = IM_low > 0
        if self.transform:
            k1 = torch.rand(1)
            k2 = torch.rand(1)
            if k1 > self.prot:
                CT = np.rot90(CT, k=1, axes=(1, 2)).copy()
                IM_low = np.rot90(IM_low, k=1, axes=(1, 2)).copy()
                IM_high = np.rot90(IM_high, k=1, axes=(1, 2)).copy()
                
            elif k1 > 2*self.prot/3:
                CT = np.rot90(CT, k=2, axes=(1, 2)).copy()
                IM_low = np.rot90(IM_low, k=2, axes=(1, 2)).copy()
                IM_high = np.rot90(IM_high, k=2, axes=(1, 2)).copy()
                     
            elif k1 > self.prot/3:
                CT = np.rot90(CT, k=3, axes=(1, 2)).copy()
                IM_low = np.rot90(IM_low, k=3, axes=(1, 2)).copy()
                IM_high = np.rot90(IM_high, k=3, axes=(1, 2)).copy()
                

            if k2 > self.pflip:
                CT = np.flip(CT, 2).copy()
                IM_low = np.flip(IM_low, 2).copy()
                IM_high = np.flip(IM_high, 2).copy()
                
            elif k2 > self.pflip/2:
                CT = np.flip(CT, 1).copy()
                IM_low = np.flip(IM_low, 1).copy()
                IM_high = np.flip(IM_high, 1).copy()
                
            

        IM_low = np.log1p(IM_low)
        IM_high = np.log1p(IM_high)

        CT = torch.from_numpy(CT).type(torch.FloatTensor)
        IM_low = torch.from_numpy(IM_low).type(torch.FloatTensor)
        IM_high = torch.from_numpy(IM_high).type(torch.FloatTensor)



        return CT, IM_low, IM_high 
