import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import os
import imageio
import cv2
import numpy as np
import pydicom
import SimpleITK
import re
from PIL import Image


def nii_img(path,output):
    for name in os.listdir(path):
        patient=os.path.join(path,name)
        example_filename = patient+'/T2W/Untitled.nii.gz'

        img = nib.load(example_filename)
        # print(img)
        # print(img.header['db_name'])  # 输出头信息
        #显示图像
        # width, height, queue= img.dataobj.shape
        # OrthoSlicer3D(img.dataobj).show()
        nii = img.get_fdata()
        # nii = np.transpose(nii, (2, 1, 0))
        path_output=output+str(name)+'/'
        os.makedirs(path_output,exist_ok=True)
        (x, y, z) = nii.shape
        for i in range(z):  # z是图像的序列
            silce = nii[:, :, i]  # 选择哪个方向的切片都可以
            imageio.imwrite(os.path.join(path_output, '{}.png'.format(i)), silce)


def nii(path,output):
    img_nii=[]
    index=0
    for name in os.listdir(path):
        file = {}
        path_input = output + str(name)
        files = os.listdir(path_input)
        for i in files:

            nii=cv2.imread(path_input+'/'+ str(i))
            i = int(str(i).replace('.png', ''))
            hsv = cv2.cvtColor(nii, cv2.COLOR_BGR2HSV)
            low_hsv = np.array([0, 0, 150])#150
            high_hsv = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
            pixels = np.asarray(mask)

            file.update({i:pixels})

        d=sorted(file.items(), key=lambda x: x[0], reverse=False)

            #
        # img_nii.append(pixels)
        for k in d:
            index+=1
            # img = np.rot90(k[1])
            # img = cv2.flip(img, 0)
            img_nii.append(k[1])
            # imageio.imwrite(os.path.join('E:/fly/data/brain/mask', str(name) + str(name) + '{}.png'.format(index)), k[1])
    return img_nii


def dcm(path):
    imgs = []
    for name in os.listdir(path):
        files = {}
        patient=os.path.join(path,name)
        for file in os.listdir(os.path.join(patient, 'T2W')):
                if file.find('.dcm') >= 0 or file.find('.DCM') >= 0:
                    img_path = os.path.join(patient + '\T2W', file)
                    file = int(re.findall('\d+', file)[0])
                    files.update({img_path: file})
        d = sorted(files.items(), key=lambda x: x[1], reverse=False)

        for i in d:
            dcm = pydicom.read_file(i[0], force=True)
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            img = dcm.pixel_array.transpose()
            img = dcm.pixel_array / np.max(dcm.pixel_array) * 255
            img = np.uint8(img)
            # imageio.imwrite(os.path.join('E:/fly/data/brain/T1ce/dcm/zl-wild/', str(name) + '{}.png'.format(i[1])), img)
            imgs.append(img)
    return imgs

# imageio.imwrite(os.path.join(path_dcm, '{}.png'.format(i[1])), img)

path = 'E:/fly/data/brain/data_ori/SJ/21'
output='E:/fly/data/brain/T2ce/mask_ori/sj-21/'
patch_path='E:/fly/data/brain/bingzao/t2/ori/sj-21/'

# nii_img(path,output)
img_dcm=dcm(path)
img_nii=nii(path,output)

for i in range(len(img_dcm)):
    threshold_level = 20
    coords = np.column_stack(np.where(img_nii[i] > threshold_level))

    if len(coords) == 0:
        continue
    else:

        list_0 = []
        list_1 = []
        for k in range(coords.shape[0]):
            # print(coords[i][0])
            list_0.append(coords[k][0])
            list_1.append(coords[k][1])

        max_0 = max(list_0)
        min_0 = min(list_0)
        max_1 = max(list_1)
        min_1 = min(list_1)
        # min_1=min_1-10
        # min_0=min_0-10
        # max_1=max_1+10
        # max_0=max_0+10

        patch = img_dcm[i][min_1:max_1, min_0:max_0]
        print(i)
        os.makedirs(patch_path,exist_ok=True)
        imageio.imwrite(os.path.join(patch_path, '{}.png'.format(i)), patch)

# for i in range(len(img_dcm)):
#
#     threshold_level = 10
#     coords = np.column_stack(np.where(img_nii[i] > threshold_level))
#     # print(i)
#     if len(coords) == 0:
#         continue
#     else:
#         # print(i)
#
#         # img = np.rot90(img_nii[i])
#         # img = cv2.flip(img, 0)
#         patch = np.multiply(img_dcm[i],img_nii[i])
#         patch=patch/255
#         imageio.imwrite(os.path.join(patch_path, '{}.png'.format(i)), patch)


# img_nii=cv2.imread('E:/fly/data/brain/T1ce/mask_ori/zl-19/0lijincheng/9.png')
# img_dcm=cv2.imread('E:/fly/data/brain/T1ce/dcm/0lijincheng9.png')
#

# patch = np.multiply(img_dcm,img_nii)
#
# plt.imshow(patch)
# plt.show()

