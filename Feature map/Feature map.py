# Selvaraju, R.R., Cogswell, M., Das, A. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. Int J Comput Vis 128, 336–359 (2020). https://doi.org/10.1007/s11263-019-01228-7

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2

from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# model1 = ResNet50(weights='imagenet', include_top=True)
model1 = ResNet50(weights=r'..\example\models\resnet50_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
#model1.summary()

def image_processing(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  
    return x


    
def gradcam_fm(model, x):
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])    
    pred_output = model.output[:, pred_class]
    
    last_conv_layer = model.get_layer('res5c_branch2c')
    
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    pooled_grads = K.sum(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])

    heatmap = np.sum(conv_layer_output_value, axis=-1)
        
    return heatmap

import matplotlib.ticker as ticker
def fmt1(x,pos):   # 设置colorbar的刻度值
    return int(x/253)
def visual_heatmap(heatmap, img_path):
    fig, ax = plt.subplots()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)


    ax.imshow(im, alpha=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    h= ax.imshow(heatmap, cmap='rainbow', alpha=0.4)

    plt.colorbar(h, ticks=[0,253],format=ticker.FuncFormatter(fmt1),shrink = 0.8)
    plt.show()



img_path = '../data/2.png'#'../example/example_roi.png' #'../data/4.png'
img = image_processing(img_path)
heatmap = gradcam_fm(model1, img)
visual_heatmap(heatmap, img_path)