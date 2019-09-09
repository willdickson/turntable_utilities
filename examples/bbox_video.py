import turntable_utilities.arena_tools as arena_tools
import turntable_utilities.fly_tools as fly_tools

in_filename = 'data/IMG_0957.mp4'
out_filename = 'bbox.avi'
#filename = 'data/2019_8_27_14_19_test_Trim.mp4'
arena_dict = arena_tools.find_arena(in_filename, bndry_threshold=0.55, erode_kernel_size=21)
fly_results = fly_tools.create_bbox_video(in_filename, out_filename, arena_dict, threshold=150, bbox_size=(80,80), display=True)
