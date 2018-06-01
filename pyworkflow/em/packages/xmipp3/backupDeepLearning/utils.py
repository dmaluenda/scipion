
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import numpy as np
import time
from skimage.transform import resize
import cv2

def computeMatrices(image, conv_output, conv_grad, gb_viz):
    
  output = conv_output 
  grads_val = conv_grad 
  print(output.shape, grads_val.shape)
  
  weights = np.mean(grads_val, axis = (0, 1))

  cam = np.ones(output.shape[0 : 2], dtype = np.float32)

  # Taking a weighted average
  print(weights.shape)
  for i, w in enumerate(weights):
    cam += w * output[:, :, i]

  # Passing through ReLU
  cam = np.maximum(cam, 0)
  cam = (cam - np.min(cam)) / (np.max(cam)- np.min(cam)) # scale 0 to 1.0    

  # print(cam)
  gb_viz -= np.min(gb_viz)
  gb_viz /= gb_viz.max()
  
  img = image.astype(float)    
  img -= np.min(img)
  img /= img.max()
  # print(img)
  cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
#  cam_heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
  # cam = np.float32(cam) + np.float32(img)
  # cam = 255 * cam / np.max(cam)
  # cam = np.uint8(cam)
  gd_gb= gb_viz[:, :, 0] * resize(cam, gb_viz[:, :, 0].shape)
  return img, cam_heatmap, gb_viz, gd_gb
    
def visualize(i, y_pred, labels, logits, image, dataClass0, dataClass1):
 
  conv_output_0, conv_grad_0, gb_viz_0=  dataClass0
  conv_output_1, conv_grad_1, gb_viz_1=  dataClass1  

  conv_output_0, conv_grad_0, gb_viz_0= conv_output_0[i,...], conv_grad_0[i,...], gb_viz_0[i,...]
  conv_output_1, conv_grad_1, gb_viz_1= conv_output_1[i,...], conv_grad_1[i,...], gb_viz_1[i,...]

  
  img, cam_0, gb_viz_0, gd_gb_0= computeMatrices(image, conv_output_0, conv_grad_0, gb_viz_0)
  __ , cam_1, gb_viz_1, gd_gb_1= computeMatrices(image, conv_output_1, conv_grad_1, gb_viz_1)
  
  fig, axs = plt.subplots(2, 4, figsize=(10, 5),squeeze=False)
  axs[0][0].imshow(np.squeeze(img), cmap='gray')
  axs[0][0].set_title('Input Image. Label= %d Score= %.2f'%(labels[1], y_pred[1] ))
  
  axs[0][1].imshow(cam_0)
  axs[0][1].set_title('Grad-CAM-class0 %.3f %.3f'%(logits[0], logits[1]))
  
  axs[1][1].imshow(cam_1)
  axs[1][1].set_title('Grad-CAM-class1')  

#  axs[0][2].imshow(np.squeeze(gb_viz), cmap='gray')
  axs[0][2].imshow(np.squeeze(gb_viz_0))
  axs[0][2].set_title('guided backpropagation-class0')
  axs[1][2].imshow(np.squeeze(gb_viz_1))
  axs[1][2].set_title('guided backpropagation-class1')

#  axs[0][3].imshow(np.squeeze(gd_gb), cmap='gray')
  axs[0][3].imshow(np.squeeze(gd_gb_0))
  axs[0][3].set_title('guided Grad-CAM-class0')
  axs[1][3].imshow(np.squeeze(gd_gb_1))
  axs[1][3].set_title('guided Grad-CAM-class1')
  plt.show()
  

