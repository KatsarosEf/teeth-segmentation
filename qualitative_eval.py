import os
import cv2
import numpy as np
import pickle

def warpVis(img1, warped):
    '''
    :param img1: numpy reference image
    :param warped: numpy warped image
    :return: warping (Red & Cyan) visualization
    '''
    warp_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    z = im_gray, im_gray, warp_gray
    return np.stack(z, axis=2)


x_prv = '/media/efklidis/4TB/dblab_real/test/b2677b41-7cea-4181-ab09-43f60bf5ebc1/input/64.jpg'
x_curr = '/media/efklidis/4TB/dblab_real/test/b2677b41-7cea-4181-ab09-43f60bf5ebc1/input/66.jpg'
# h = '/media/efklidis/4TB/results/mia-paper/MOST-NSS++ | ATB/1f3041c2-2971-49eb-95e0-ced6f46e7b6e/homos2.npy'
h_gt = '/media/efklidis/4TB/dblab_real/test/b2677b41-7cea-4181-ab09-43f60bf5ebc1/homo_4df.pkl'
with open(h_gt, 'rb') as homo_pickle:
    h_gt = pickle.load(homo_pickle)


index = 1

img_prv = cv2.imread(x_prv)
img_curr = cv2.imread(x_curr)
warped_prv = warpVis(img_prv, img_curr)
cv2.imwrite('./unaligned66.png', warped_prv)

# homo = np.load(h)[index]
homo_gt = h_gt[0:index+1]
chain = np.eye(3)
for i in range(len(homo_gt)):
    chain = np.matmul(chain, homo_gt[i])


img_prv = cv2.warpPerspective(img_curr, np.linalg.inv(chain), (img_prv.shape[1], img_prv.shape[0]))
warped_prv = warpVis(img_curr, img_prv)
cv2.imwrite('./aligned66.png', warped_prv)



