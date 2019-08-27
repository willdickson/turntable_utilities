import  turntable_utilities
import numpy as np
import cv2

filename = 'data/IMG_0957.mp4'
results = turntable_utilities.arena_tools.find_arena(filename)

center_pt = results['center_pt']
mask_gray = results['mask_gray']
mask_bgr = results['mask_bgr']
img_max = results['img_max']
img_w_contour = results['img_w_contour']

test_img_gray = 255*np.ones(mask_gray.shape,dtype=np.uint8)
test_img_gray[mask_gray] = 0


test_img_bgr = 255*np.ones(mask_bgr.shape,dtype=np.uint8)
test_img_bgr[mask_bgr] = 0

cv2.imshow('img_max', img_max)
cv2.imshow('mask_bgr', test_img_bgr)
cv2.imshow('mask_gray', test_img_gray)
cv2.imshow('img_w_contour', img_w_contour)
cv2.waitKey(-1)
