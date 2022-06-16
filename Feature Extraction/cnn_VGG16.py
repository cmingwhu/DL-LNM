import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
from keras.preprocessing import image



## Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# model = VGG16(weights='imagenet', include_top=False)
model = VGG16(weights=r'..\example\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

model.summary()


# Load batch file
imgDir = 'G:/ZHP/all'
dirlist = os.listdir(imgDir)[0:]
print(dirlist)

# read images in Nifti format 
def loadSegArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    segPath = [os.path.join(path,i) for i in pathList if ('mask' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg

# read regions of interest (ROI) in Nifti format 
def loadImgArraywithID(fold,iden):
    
    path = fold
    pathList = os.listdir(path)
    
    imgPath = [os.path.join(path,i) for i in pathList if ('mask' not in i.lower()) & (iden not in i.lower())][0]
    img = sitk.ReadImage(imgPath)    
    return img

def write_image(image_data, image_path):

    _image = sitk.GetArrayFromImage(image_data)
    _image = sitk.GetImageFromArray(_image)

    sitk.WriteImage(_image, image_path)


# Feature Extraction
#Cropping box
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)  # 返回索引
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)
        
def featureextraction(imageFilepath,maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath) 
    mask_array = sitk.GetArrayFromImage(maskFilepath)

    # write_image(image_array, '../data/roi_images11.png')


    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1, dtype=np.float)
    roi_im = sitk.GetImageFromArray(roi_images1)

    # write_image(roi_images1, '../data/roi_images11.png')
    from skimage import io, data
    import cv2
    from imageio import imread, imwrite

    # io.imsave('../data/roi_images1.png', roi_images1)
    # sitk.WriteImage(roi_im,'../data/roi_im.nii')
    cv2.imwrite("../data/4.png" , roi_images1)
    io.imsave('../data/44.png', roi_images2)
    # cv2.imwrite("../data/11.png", roi_images2)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(roi_images1)
    ax2.imshow(roi_images2)
    plt.show()


    fig = plt.figure()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(roi_images)
    ax2.imshow(roi_images1)
    plt.show()



    x = image.img_to_array(roi_images2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model_pool_features = model.predict(x)
    
    feature_map = base_model_pool_features[0]
    feature_map = feature_map.transpose((2,1,0))
    features = np.max(feature_map,-1)
    features = np.max(features,-1)
    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures
 
featureDict = {}
for ind in range(len(dirlist)):
    ind =3
    path = os.path.join(imgDir,'d7')
    # path = os.path.join(imgDir,dirlist[ind])

    seg = loadSegArraywithID(path,'mask')
    im = loadImgArraywithID(path,'mask')
        
    deeplearningfeatures = featureextraction(im,seg) 

    result = deeplearningfeatures
    key = list(result.keys())
    key = key[0:]
        
    feature = []
    for jind in range(len(key)):
        feature.append(result[key[jind]])
        
    featureDict[dirlist[ind]] = feature
    dictkey = key
    print(dirlist[ind])
    
dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
dataframe.to_csv('..\example/Features_VGG16.csv')