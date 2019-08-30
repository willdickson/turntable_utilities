import turntable_utilities

#filename = 'data/IMG_0957.mp4'
filename = 'data/2019_8_27_14_19_test_Trim.mp4'
arena_dict = turntable_utilities.arena_tools.find_arena(filename, bndry_threshold=0.55, erode_kernel_size=21)
fly_results = turntable_utilities.fly_tools.find_fly_in_video(filename, arena_dict, threshold=90, display=True)



