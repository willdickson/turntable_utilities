from __future__ import print_function
import numpy as np
import scipy
import scipy.signal
import cv2
from . import contour_tools

import matplotlib.pyplot as plt


def find_fly_in_video(filename, arena_dict, threshold=127,  head_frac=0.15, display=True, fly_vec_len=25):
    """
    Finds the position (centroid) and body angle of the fly in each frame of the video.

    Arguments:

      filename    = name of videofile to analyze. 
      arena_dict  = dictionary of arena values found by the find_arena function in arena_tools
      threshold   = threshold for finding the fly blob
      head_frac   = fraction of body len to use for finding fiting head and tail slopes.
      display     = display results and video is analyzed
      fly_vec_len = length of fly body vector shown in display in pixels

    Returns:
        
        results_dict  = diction of values with the following key
            'frames'    =  list of frame numbers 
            'angles'    =  list of fly body angles
            'centroids' =  list of x,y coordinates of fly body centroids

    """


    cap = cv2.VideoCapture(filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_list = []
    angle_list = []
    centroid_list = []
    
    while True:
        
        ret, img_bgr = cap.read()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break
    
        # Apply boundary mask
        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        img_gray[arena_dict['mask_gray']] = 255
    
        # Find fly in image and extract relevant data e.g. centroid, body vector, etc.
        angle, angle_data = find_fly_in_image(img_gray, threshold, 1.0)
        fly_cx, fly_cy = angle_data['centroid']
        fly_vec_x, fly_vec_y = angle_data['body_vector']
    
        img_rot_thresh = angle_data['rotated_threshold_image']
        n,m = img_rot_thresh.shape
    
        # Compute the array widths of the fly along the body axis in order to try and determine
        # the head direction. Smooth widths with Savitzky Golay filter
        widths = img_rot_thresh.sum(axis=0)
        widths = scipy.signal.savgol_filter(widths,9,2,deriv=0)
        widths_ind = np.arange(widths.shape[0])

        # Find mask for nonzero width values. Also get min and max indices for nonzero values - should
        # be start and stop points for fly body. 
        nonzero_mask = widths > 0
        nonzero_widths = widths[nonzero_mask]
        nonzero_ind = widths_ind[nonzero_mask]
        nonzero_min = nonzero_ind.min()
        nonzero_max = nonzero_ind.max()
        nonzero_ind = nonzero_ind - m/2  
        num_pts = int(head_frac*nonzero_widths.shape[0])

        # Get positive (front) part of nonzero body widths and reverse order
        nonzero_widths_pos = nonzero_widths[-num_pts:]
        nonzero_widths_pos = nonzero_widths_pos[::-1]
        nonzero_ind_pos = np.arange(num_pts)
    
        # Get negative (back) part of nonzero body widths
        nonzero_widths_neg = nonzero_widths[:num_pts]
        nonzero_ind_neg = np.arange(num_pts)

    
        # Fit lines to pos and neg body widths points - we will use slopes to find body orientation
        fit_pos = np.polyfit(nonzero_ind_pos, nonzero_widths_pos,1)
        fit_neg = np.polyfit(nonzero_ind_neg, nonzero_widths_neg,1)

        # Change orientation of fly on test
        fly_direction_test = fit_pos[0] > fit_neg[0]
        if fly_direction_test:
            fly_p0 = int(fly_cx - fly_vec_len*fly_vec_x), int(fly_cy - fly_vec_len*fly_vec_y)
            fly_p1 = int(fly_cx + fly_vec_len*fly_vec_x), int(fly_cy + fly_vec_len*fly_vec_y)
        else:
            fly_p0 = int(fly_cx + fly_vec_len*fly_vec_x), int(fly_cy + fly_vec_len*fly_vec_y)
            fly_p1 = int(fly_cx - fly_vec_len*fly_vec_x), int(fly_cy - fly_vec_len*fly_vec_y)
            angle = -angle
        angle = np.rad2deg(angle)
    
        print('{}/{}, angle: {:0.1f}'.format(frame_number,frame_count,angle))
    
        frame_list.append(frame_number)
        angle_list.append(angle)
        centroid_list.append((fly_cx,fly_cy))
    
        # Optional display - shows results during analysis 
        if display:
            fly_center = int(fly_cx), int(fly_cy)
            cv2.circle(img_bgr, fly_center, 2, (255,0,0), 5)
            cv2.arrowedLine(img_bgr, fly_p0, fly_p1, (0,0,255), 2)

            cv2.imshow('original',img_bgr)
            cv2.imshow("angle_data['contour_image']", angle_data['contour_image'])
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    if display:
        cap.release()
        cv2.destroyAllWindows()

    results_dict = {
            'frames'    :  frame_list,
            'angles'    :  angle_list,
            'centroids' : centroid_list,
            }

    return results_dict





def find_fly_in_image(image, threshold=60, mask_scale=0.95): 
    """
    Finds the the fly in a given an image.  Fly should be dark against a light background.

    Parameters:

      threshold   = value of threshold used to separate fly's from the (light) image background.

      mask_scale  = scales the radius of the circular mask around the fly's centroid. The value 
                    Should be in range [0,1]. 1 means the radius = min(image_width, image_height) 

    Returns: tuple (angle, angle_data) where

      angle      = the estimate of the fly's angle in radians

      angle_data = dictionary of useful data and images calculated during the angle estimation

      angle_data = { 
          'moments': the moments of the maximum area contour in the thresholded image,
          'centroid':  (x,y) coordinates for the centroid of the fly (thresholded fly blob really),
          'max_contour': the maximum area contour,
          'max_contour_area':  the area of the maximum area contour,
          'body_vector': unit vector along the fly's body axis,
          'contour_image': image with body contour, centroid, and fly's body axis drawn on it,
          'shifted_image': image shifted so that the fly's cendroid is at the center,
          'rotated_image': image rotated by the fly's body angle an shift so centroid is centered,
          'threshold_image': the thresholded image,
          'shifted_threshold_image': thresholded image shifted so the centroid is centered,
          'rotated_threshold_image': thresholded image rotated and shifted 
      }

    """

    # Get basic image data
    height, width = image.shape
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height

    # Threshold, find contours and get contour with the maximum area
    rval, threshold_image = cv2.threshold(image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY_INV)
    dummy, contour_list, dummy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour, max_area = contour_tools.get_max_area_contour(contour_list)

    # Get moments and then compute centroid and rotation angle
    moments = cv2.moments(max_contour)
    centroid_x, centroid_y = contour_tools.get_centroid(moments)
    angle, body_vector = get_angle_and_body_vector(moments)

    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)

    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(centroid_x + axis_length*body_vector[0]), int(centroid_y + axis_length*body_vector[1])
    body_axis_pt_1 = int(centroid_x - axis_length*body_vector[0]), int(centroid_y - axis_length*body_vector[1])

    # Compute cirlce mask
    mask_radius = int(mask_scale*height/2.0)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2

    # Draw image with body contours, centroid circle and body axis
    centroid = int(centroid_x), int(centroid_y)
    contour_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image,[max_contour],-1,(0,0,255),2)

    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - centroid_x)], [0.0, 1.0, (mid_y - centroid_y)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(image, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(threshold_image, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = rotated_threshold_image*circ_mask
    rval, rotated_threshold_image = cv2.threshold(rotated_threshold_image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY)

    data = {
            'moments': moments,
            'centroid': centroid,
            'max_contour': max_contour,
            'max_contour_area':  max_area,
            'body_vector': body_vector,
            'contour_image': contour_image,
            'shifted_image': shifted_image,
            'rotated_image': rotated_image,
            'threshold_image': threshold_image,
            'shifted_threshold_image': shifted_threshold_image,
            'rotated_threshold_image': rotated_threshold_image,
            }

    return angle, data 
        


def get_angle_and_body_vector(moments): 
    """
    Computre the angle and body vector given the image/blob moments
    """
    body_cov = np.array( [ [moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02'] ]])
    eig_vals, eig_vecs = np.linalg.eigh(body_cov)
    max_eig_ind = np.argmax(eig_vals**2)
    max_eig_vec = eig_vecs[:,max_eig_ind]
    angle = np.arctan2(max_eig_vec[1], max_eig_vec[0])
    return angle, max_eig_vec



def create_bbox_video(in_filename, out_filename, arena_dict, threshold=127, bbox_size=(100,100), display=True):
    """
    """

    cap = cv2.VideoCapture(in_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_filename, fourcc, 30.0, bbox_size)


    frame_list = []
    angle_list = []
    centroid_list = []
    
    while True:
        
        ret, img_bgr = cap.read()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break
    
        # Apply boundary mask
        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
        img_gray[arena_dict['mask_gray']] = 255
    
        # Find fly in image and extract relevant data e.g. centroid, body vector, etc.
        angle, angle_data = find_fly_in_image(img_gray, threshold, 1.0)
        fly_cx, fly_cy = angle_data['centroid']
        fly_vec_x, fly_vec_y = angle_data['body_vector']
    
        #img_rot_thresh = angle_data['rotated_threshold_image']
        #n,m = img_rot_thresh.shape

        fly_center = int(fly_cx), int(fly_cy)
        p = int(fly_center[0] - 0.5*bbox_size[0]), int(fly_center[1] - 0.5*bbox_size[1])
        q = int(fly_center[0] + 0.5*bbox_size[0]), int(fly_center[1] + 0.5*bbox_size[1])
        img_cropped = img_gray[p[1]:q[1],p[0]:q[0]]
        img_out = cv2.cvtColor(img_cropped,cv2.COLOR_GRAY2BGR)
        out.write(img_out)
    
        # Optional display - shows results during analysis 
        if display:
            cv2.circle(img_bgr, fly_center, 2, (255,0,0), 5)
            cv2.rectangle(img_bgr, p, q, (0,0,255), 2)  
            cv2.imshow('original',img_bgr)
            cv2.imshow('cropped', img_cropped)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    out.release()
    if display:
        cap.release()
        cv2.destroyAllWindows()



# ---------------------------------------------------------------------------------------
if __name__ == '__main__':

    import arena_tools

    filename = 'IMG_0957.mp4'
    arena_dict = arena_tools.find_arena(filename, bndry_threshold=200, erode_kernel_size=13)
    fly_results = find_fly_in_video(filename, arena_dict, threshold=127, display=True)





