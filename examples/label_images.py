from __future__ import print_function
import sys
import cv2
import h5py
import numpy as np

# Key maps
KEY_QUIT = ord('q')
KEY_NEXT = ord('n')
KEY_PREV = ord('p')
KEY_FIRST = ord('f')
KEY_LATEST = ord('e')
KEY_LOCK = ord(' ')
KEY_UNLOCK = ord('u')
KEY_SAVE = ord('s')

KEY_L = ord('h')
KEY_R = ord('l') 
KEY_U = ord('k') 
KEY_D = ord('j') 

LABEL_MOVE_KEY_LIST = [KEY_L, KEY_R, KEY_U, KEY_D]
LABEL_MOVE_TABLE = {
        KEY_L: (-1, 0),
        KEY_R: ( 1, 0),
        KEY_U: ( 0,-1),
        KEY_D: ( 0, 1),
        }


class LabelMaker(object):


    def __init__(self, input_filename, output_filename, propagate=True):
        self.input_filename = input_filename
        self.output_filename = output_filename

        self.cap = cv2.VideoCapture(self.input_filename)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)

        self.propagate = propagate
        self.locked = False
        self.frame = 0

        self.label_dict = {}
        self.image_dict = {}

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.label_dict[self.frame] = (x,y)
            print('frame: {}, point: ({},{})'.format(self.frame,x,y))
            self.update_frame()

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frame)
        ret, img_bgr = self.cap.read()
        if not ret:
            return
        if not self.image_dict.has_key(self.frame):
            self.image_dict[self.frame] = img_bgr
        img_bgr_copy = np.array(img_bgr)
        try:
            label = self.label_dict[self.frame]
        except KeyError:
            label = None
        
        if label is not None:
            cv2.circle(img_bgr_copy, label, 2, (0,0,255), 1)
        cv2.imshow('image', img_bgr_copy)

    def run(self):

        while True:
            print('fame: {}'.format(self.frame))
            self.update_frame()
            key = cv2.waitKeyEx(0) & 0xFF
            #print(key)
            if key ==  KEY_QUIT:
                break
            elif key == KEY_NEXT:
                if not (self.locked and not self.label_dict.has_key(self.frame+1)):
                    self.frame += 1
                    if not self.label_dict.has_key(self.frame) and self.label_dict.has_key(self.frame-1):
                        self.label_dict[self.frame] = self.label_dict[self.frame -1]
                        print('new label!')
            elif key == KEY_PREV:
                self.frame -= 1
                self.frame = max(0,self.frame)
            elif key == KEY_FIRST:
                self.frame = 0
            elif key == KEY_LATEST:
                keys = self.label_dict.keys()
                keys.sort()
                self.frame = keys[-1]
            elif key == KEY_LOCK:
                self.locked = True
                print('locked')
            elif key == KEY_UNLOCK:
                self.locked = False
                print('unlocked')
            elif key == KEY_SAVE:
                self.save_labels()
            elif key in LABEL_MOVE_KEY_LIST:
                try:
                    label = self.label_dict[self.frame]
                except KeyError:
                    label = None
                if label is not None:
                    x, y = label
                    dx, dy = LABEL_MOVE_TABLE[key]
                    self.label_dict[self.frame] = x + dx, y + dy


    def save_labels(self):
        image_list = []
        label_list = []
        frame_list = []
        for frame in sorted(self.image_dict.keys()):
            image = self.image_dict[frame]
            try:
                label = self.label_dict[frame]
            except KeyError:
                continue
            frame_list.append(frame)
            image_list.append(image)
            label_list.append(label)
        frame_array = np.array(frame_list)
        image_array = np.array(image_list)
        label_array = np.array(label_list)
        print('Saving {} labeled frames'.format(frame_array.shape[0]))
        h5file = h5py.File(self.output_filename, 'w')
        h5file.create_dataset('frame',data=frame_array)
        h5file.create_dataset('image',data=image_array)
        h5file.create_dataset('label',data=label_array)

        h5file.close()






# ---------------------------------------------------------------------------------------
if __name__ == '__main__':

    input_filename = sys.argv[1]
    output_filename = 'labels.h5'

    label_maker = LabelMaker(input_filename,output_filename)
    label_maker.run()




