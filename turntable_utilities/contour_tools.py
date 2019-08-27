import cv2
import numpy as np

def get_max_area_contour(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and 
    returns 
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    max_area = contour_areas.max()
    max_ind = contour_areas.argmax()
    max_contour = contour_list[max_ind]
    return max_contour, max_area


def get_centroid(moments): 
    """
    Computer centroid given the image/blob moments
    """
    if moments['m00'] > 0:
        centroid_x = int(np.round(moments['m10']/moments['m00']))
        centroid_y = int(np.round(moments['m01']/moments['m00']))
    else:
        centroid_x = 0
        centroid_y = 0
    return centroid_x, centroid_y
