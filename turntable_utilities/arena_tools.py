from __future__ import print_function
import cv2
import numpy as np
import contour_tools


def find_arena(filename, bndry_threshold=200, erode_kernel_size=13):
    """ 
    Finds center of rotation and boundary of the rotating arena. 

    Arguments:

      filename          = name of video file
      bndry_threshold   = threshold used to detect the boundary
      erode_kernel_size = size of erosion ernel to used to shrink mask and remove boundary artifacts.

    Returns:

        arean_dict = dictionary arena values extacted from video with the following keys
          'center_pt'    = (cx,cy) center of rotation for arena
          'mask_gray'    = arena boundary mask for gray images 
          'mask_bg'      = arena boundary mask for bgr images
          'img_max'      = image whose pixles have maximal value over al frame is video
          'img_w_contour = image showing with boundary contour and center point 

    """
    cap = cv2.VideoCapture(filename)

    # Loop over frames and find image whose pixel values are max value for all frames
    img_max = None
    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break
        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        if img_max is None:
            img_max = img_gray
        else:
            img_max = cv2.max(img_max,img_gray)
    
    # Threshold max image and then find boundary contour
    ret, img_thresh = cv2.threshold(img_max, bndry_threshold, 255, 0)
    _, contour_list, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour, max_area = contour_tools.get_max_area_contour(contour_list)
    
    # Find the center point of the arena. Note, this should coincide with the rotation center assuming 
    # the arena was rotating > 360 degrees in video
    moments = cv2.moments(max_contour)
    cx, cy = contour_tools.get_centroid(moments)
    center_pt = cx,cy
    
    # Create image which display boundary contour and center point 
    img_w_contour = cv2.cvtColor(img_max, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_w_contour,[max_contour],-1,(0,0,255),2)
    cv2.circle(img_w_contour, center_pt, 2, (0,255,0), 2)
    
    # Shrink mask using erosion operating. This is done to eliminate any artifacts at the boundary of
    # arena as the actual arena isn't a perfect circle.
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erode_kernel_size,erode_kernel_size))
    img_thresh_erode = cv2.erode(img_thresh,erode_kernel,iterations=1)

    # Create masks 
    mask_gray = img_thresh_erode < 127
    mask_bgr = np.expand_dims(mask_gray,3)
    mask_bgr = mask_bgr.repeat(3,axis=2)

    results_dict = {
            'center_pt'     : center_pt, 
            'mask_gray'     : mask_gray,
            'mask_bgr'      : mask_bgr,
            'img_max'       : img_max, 
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




