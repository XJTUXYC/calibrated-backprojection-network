import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.utils import flow_to_image
import data_utils

img1 = Image.open('/home/xyc/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png').convert('RGB')
img1 = img1.resize((768, 304))
img1 = np.asarray(img1, np.float32)
img1 = np.transpose(img1, (2, 0, 1))
img1 = img1 / 255.0
img1 = torch.from_numpy(img1)
img1 = torch.unsqueeze(img1, 0)

img2 = Image.open('/home/xyc/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png').convert('RGB')
img2 = img2.resize((768, 304))
img2 = np.asarray(img2, np.float32)
img2 = np.transpose(img2, (2, 0, 1))
img2 = img2 / 255.0
img2 = torch.from_numpy(img2)
img2 = torch.unsqueeze(img2, 0)

flow_model = models.optical_flow.raft_small(pretrained=True)
flow_model.eval()
# time_start  = time.time()
flow12 = flow_model.forward(img1, img2)[-1]
# time_end  = time.time()
# print('time cost flow',1000*(time_end-time_start),'ms')
print(flow12[:, :, 20, 750])
flow_img = flow_to_image(flow12)
data_utils.plot_flow([img1[0], flow_img[0]])

img1 = cv2.imread ('/home/xyc/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png')
img1 = cv2.resize(img1, (768, 304), interpolation=cv2.INTER_AREA)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = np.transpose(img1, (2, 0, 1))
img2 = cv2.imread('/home/xyc/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png')
img2 = cv2.resize(img2, (768, 304), interpolation=cv2.INTER_AREA)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# time_start  = time.time()
flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# time_end  = time.time()
# print('time cost flow',1000*(time_end-time_start),'ms')
flow = np.transpose(flow, (2, 0, 1))

flow = torch.from_numpy(flow)
print(flow[:, 20, 750])
flow_img = flow_to_image(flow)
img1 = torch.from_numpy(img1)/255
data_utils.plot_flow([img1, flow_img])
plt.show()