import  turntable_utilities

filename = 'data/IMG_0957.mp4'
results = turntable_utilities.arena_tools.find_arena(filename)

center_pt = results['center_pt']
mask_gray = results['mask_gray']
mask_bgr = results['mask_bgr']
img_max = results['img_max']
img_w_contour = results['img_w_contour']
