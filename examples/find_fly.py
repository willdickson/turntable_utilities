import turntable_utilities

filename = 'data/IMG_0957.mp4'
arena_dict = turntable_utilities.arena_tools.find_arena(filename, bndry_threshold=200, erode_kernel_size=13)
fly_results = turntable_utilities.fly_tools.find_fly_in_video(filename, arena_dict, threshold=127, display=True)
