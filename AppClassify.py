import os
import glob
import os
import time
from shutil import rmtree

import cv2
import numpy as np
from keras.models import load_model

from BaseVGG19Extractor import VGG19Extractor
from VGG19FeatureExtractor import RescaleList
from project_utils import EnsurePath

COUGHING_MODEL = '/Users/pprusty05/workspace/Deep_learning/learned_model/coughing_vgg/model_dir/entire_model.h5'
SNEEZING_MODEL = '/Users/pprusty05/workspace/Deep_learning/learned_model/sneezing_vgg/model_dir/entire_model.h5'
WAVING_MODEL = '/Users/pprusty05/workspace/Deep_learning/learned_model/waving_vgg/model_dir/entire_model.h5'
FALLING_MODEL = '/Users/pprusty05/workspace/Deep_learning/learned_model/falling_vgg/model_dir/entire_model.h5'
TESTING_DIR = '/Users/pprusty05/workspace/Deep_learning/testing_tmp_dir'

test_video_list = []
#test_video_list.append('/Users/pprusty05/workspace/Deep_learning/data/Training/trimmed_videos/coughing/1.mp4')
test_video_list.append('/Users/pprusty05/google_drive/Deep_Learning/video1.mp4')

##Clean the directory
def clean_tmp_dir():
    rmtree(TESTING_DIR)
    EnsurePath(TESTING_DIR)



for sample_file in test_video_list:
    total_time = 0

    start_time = time.time()
    clean_tmp_dir()

    ##Extract images from videos
    IMG_DIR = os.path.join(TESTING_DIR, 'extracted_frames')
    EnsurePath(IMG_DIR)
    vidcap = cv2.VideoCapture(sample_file)

    # #Create the directory into for the file
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #print("Number of frames: ", video_length)
    count = 0
    #print("Converting video..\n")
    # Start converting the video
    while vidcap.isOpened():
        # Extract the frame
        ret, frame = vidcap.read()
        # Write the results back to output location.
        cv2.imwrite(IMG_DIR + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length - 1)):
            # Log the time again

            # Release the feed
            vidcap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            # print("It took %d seconds forconversion." % (time_end - time_start))
            break

    # total_time = total_time + time.time() - start_time
    ## For every images create  path = os.path.join(source_folder, numbered_folder)
    EXTRACTED_FEATURES_PATH = os.path.join(TESTING_DIR, 'feature')
    images = sorted(glob.glob(os.path.join(IMG_DIR, '*jpg')))
    frames = RescaleList(images, 60)
    #frames = images
    model = VGG19Extractor()
    print('VGG19 model loading is done')
    # start_time = time.time()
    if frames is not None:
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)
        # Save the sequence.
        # np.save(EXTRACTED_FEATURES_PATH, sequence)
    print('npy file extracted')
    # total_time = total_time + time.time() - start_time


    ##Prepare X
    X = []
    X.append(sequence)
    # X.append(np.load(EXTRACTED_FEATURES_PATH + '.npy'))
    X = np.array(X)
    ##Now load the model
    lstm_coughing_model = load_model(COUGHING_MODEL)
    print('Coughing model loading is done')
    # start_time = time.time()
    y1 = lstm_coughing_model.predict(X)
    # total_time = total_time + time.time() - start_time
    lstm_falling_model = load_model(FALLING_MODEL)
    print('Falling model loading is done')
    # start_time = time.time()
    y2 = lstm_falling_model.predict(X)
    # total_time = total_time + time.time() - start_time
    lstm_sneezing_model= load_model(SNEEZING_MODEL)
    print('Sneezing model loading is done')
    # start_time = time.time()
    y3 = lstm_sneezing_model.predict(X)
    # total_time = total_time + time.time() - start_time
    lstm_waving_model = load_model(WAVING_MODEL)
    print('Waving model loading is done')
    # start_time = time.time()
    y4 = lstm_waving_model.predict(X)
    total_time = total_time + time.time() - start_time
    print(y1)
    print(y2)
    print(y3)
    print(y4)

    print('Time ')
    print(total_time)



