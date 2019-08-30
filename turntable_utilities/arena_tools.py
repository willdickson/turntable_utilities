from __future__ import print_function
import cv2
import sys
import numpy as np
import contour_tools

# DEVEL TMP
import matplotlib.pyplot as plt


def find_arena(
        filename,  
        method='median', 
        frame_step=1, 
        bndry_threshold=0.5, 
        blur_kernel_size=11, 
        erode_kernel_size=11, 
        show_transect=False
        ):
    """ 
    Finds center of rotation and boundary of the rotating arena. 

    Arguments:

      filename          = name of video file
      method            = method used to create boundry image 'median' or 'max'
      frame_step        = step size to use when processing frames with 'method' 1 = every frame, 
                          2 = every other frame, etc.
      bndry_threshold   = [0,1] threshold specifying pixel intensity used to detect the boundary. 
                          0 => min intensity in image, 1 => max intensity in image
      blur_kernel_size  = size of bluring kernel used to remove noise (None for no bluring)
      erode_kernel_size = size of erosion ernel to used to shrink mask and remove boundary artifacts.
                          (None for no erosion)
      show_transect     = flag True/False indicating whether or not to show image transects 

    Returns:

        arean_dict = dictionary arena values extacted from video with the following keys
          'center_pt'    = (cx,cy) center of rotation for arena
          'mask_gray'    = arena boundary mask for gray images 
          'mask_bg'      = arena boundary mask for bgr images
          'img_max'      = image whose pixles have maximal value over al frame is video
          'img_w_contour = image showing with boundary contour and center point 

    """
    cap = cv2.VideoCapture(filename)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Loop over frames and find image whose pixel values are max value for all frames
    img_stack = []
    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break
        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        img_stack.append(img_gray)
        print('reading frame: ', len(img_stack))

    print('processing ...', end='')
    sys.stdout.flush()
    img_stack = np.array(img_stack)
    if method == 'max':
        img_bndry = img_stack[::frame_step,:,:].max(axis=0)
    else:
        img_bndry = np.median(img_stack[::frame_step,:,:], axis=0)
        img_bndry = img_bndry.astype(np.uint8)
    print('done')
    if blur_kernel_size is not None:
        #img_bndry = cv2.blur(img_bndry,(15,15))
        #img_bndry = cv2.GaussianBlur(img_bndry,(blur_kernel_size,blur_kernel_size),0)
        img_bndry = cv2.medianBlur(img_bndry,blur_kernel_size)
    
    # Threshold bndry image and then find boundary contour
    max_val = img_bndry.max()
    min_val = img_bndry.min()
    bndry_threshold = int(float(min_val) + bndry_threshold*(float(max_val) - float(min_val)))

    n,m = img_bndry.shape
    n0 = int(0.4*n)
    n1 = int(0.6*n)
    for i in range(n0,n1):
        x = img_bndry[i,:]
        plt.plot(x,'.b')
    plt.plot([0,m],[bndry_threshold, bndry_threshold],'r')
    plt.plot([0,m],[max_val, max_val],'g')
    plt.plot([0,m],[min_val, min_val],'g')
    plt.xlabel('pixel x coord')
    plt.ylabel('intensity')
    plt.title('image transect')
    plt.show()

    ret, img_thresh = cv2.threshold(img_bndry, bndry_threshold, 255, 0)
    _, contour_list, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour, max_area = contour_tools.get_max_area_contour(contour_list)
    
    # Find the center point of the arena. Note, this should coincide with the rotation center assuming 
    # the arena was rotating > 360 degrees in video
    moments = cv2.moments(max_contour)
    cx, cy = contour_tools.get_centroid(moments)
    center_pt = cx,cy
    
    # Create image which display boundary contour and center point 
    img_w_contour = cv2.cvtColor(img_bndry, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_w_contour,[max_contour],-1,(0,0,255),2)
    cv2.circle(img_w_contour, center_pt, 2, (0,255,0), 2)

    # Fill max contour 
    img_filled = img_thresh
    img_filled = np.zeros(img_thresh.shape)
    cv2.fillPoly(img_filled,pts=[max_contour],color=255)
    
    # Shrink mask using erosion operating. This is done to eliminate any artifacts at the boundary of
    # arena as the actual arena isn't a perfect circle.
    if erode_kernel_size is not None:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erode_kernel_size,erode_kernel_size))
        img_filled = cv2.erode(img_filled,erode_kernel,iterations=1)

    # Create masks 
    mask_gray = img_filled < 127
    mask_bgr = np.expand_dims(mask_gray,3)
    mask_bgr = mask_bgr.repeat(3,axis=2)

    results_dict = {
            'center_pt'     : center_pt, 
            'mask_gray'     : mask_gray,
            'mask_bgr'      : mask_bgr,
            'img_bndry'       : img_bndry, 
            'img_w_contour' : img_w_contour,
            }

    return results_dict 



# ---------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Test/Example

    filename = 'IMG_0957.mp4'
    results = find_arena(filename)

    center_pt = results['center_pt']
    mask_gray = results['mask_gray']
    mask_bgr = results['mask_bgr']
    img_max = results['img_max']
    img_w_contour = results['img_w_contour']




