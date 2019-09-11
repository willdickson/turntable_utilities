from __future__ import print_function
import sys
import cv2
import h5py
import numpy as np

def convert_labels_to_probs(input_filename,output_filename,kernel_size=(11,11),kernel_sigma=3.0, display=True):

    # Load labels
    h5_input_file = h5py.File(input_filename,'r')
    frame_array = np.array(h5_input_file['frame'])
    image_array = np.array(h5_input_file['image'])
    label_array = np.array(h5_input_file['label'])
    h5_input_file.close()

    # Create probability array
    probs_shape = (image_array.shape[0], image_array.shape[1], image_array.shape[2])
    probs_array = np.zeros(probs_shape,dtype=np.float64)
    for i in range(label_array.shape[0]):
        nx,ny = tuple(label_array[i,:])
        probs_array[i,ny,nx] = 1.0
        probs_array[i,:,:] = cv2.GaussianBlur(probs_array[i,:,:],kernel_size,kernel_sigma, kernel_sigma, cv2.BORDER_DEFAULT)
        if display:
            probs = (255/probs_array[i,:,:].max())*probs_array[i,:,:]
            probs = probs.astype(np.uint8)

            image = image_array[i,:,:,:] 
            image[:,:,2] = cv2.max(probs,image[:,:,2])

            cv2.imshow('image',image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
    # Save probs
    h5_output_file = h5py.File(output_filename,'w')
    h5_output_file.create_dataset('frame',data=frame_array)
    h5_output_file.create_dataset('image',data=image_array)
    h5_output_file.create_dataset('probs',data=probs_array)
    h5_output_file.close()


# ------------------------------------------------------------------------
if __name__ == '__main__':

    input_filename = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    else:
        output_filename = 'labels_prob.h5'

    convert_labels_to_probs(input_filename, output_filename)


