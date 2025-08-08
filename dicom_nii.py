import glob
import numpy as np
import pydicom as dcm
from scipy.sparse import data
from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import rotate
import natsort
import cv2
import pdb
def normalize(volume, wc=-500, ww=1500):
    max = wc + int(ww / 2)
    min = wc - int(ww / 2)

    volume[volume < min] = min
    volume[volume > max] = max
    volume=(volume-min)/(max-min)
    return volume
def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    # X = X - np.min(X)
    # X = X / np.max(X)
    # X = (X*255.0).astype('uint8')
    return X
 
# save_2d_path="/hdd1/Data/2D_image/train/"
# save_3d_path="/home/mars/workspace/cy_workspace/OMU_redata"
save_3d_path = "C:/Users/user1/Desktop/Changhee/project/nifti"
# save_2d_test_path = "/hdd1/Data/2D_image/test/"
# save_3d_test_path = "/hdd1/Data/3D_CT/test/"
#
datapath = "C:/Users/user1/Desktop/Changhee/project/data_DICOM/Normal/*" # TB, PN
names = glob.glob(datapath)
names = natsort.os_sorted(names)
# for name in tqdm(names[:int(len(names))]):
for name in tqdm(names):
    arrs = glob.glob(name + '/*')
    select = 10000
    selectdata = None
    files = []
    for dt in natsort.os_sorted(glob.glob(name + '/*')):
        files.append(dcm.dcmread(dt))

    slices = natsort.os_sorted(files, key=lambda s: s.ImagePositionPatient[-2])

    matrixs = []
    for dat in slices:
        matrixs.append(dat.pixel_array)
    s = int(files[0].RescaleSlope)
    b = int(files[0].RescaleIntercept)
    matrixs = np.array(matrixs).astype(np.float64)
    dataarray = (s * matrixs) + b
    dataarray[dataarray<-1024] = -1024
    dataarray = window(dataarray, WL=-20, WW=2048)
    # cv2.imwrite(save_2d_path +'/'+name.split('/')[-1]+'.png',saved)
    # dataarray = dataarray.transpose(2,1,0)
    # dataarray = resize(dataarray,(512,512,dataarray.shape[2]))
    dataarray=nib.Nifti1Image(dataarray,None)
    nib.save(dataarray,save_3d_path+'/'+name.split('/')[-1]+'.nii.gz')

