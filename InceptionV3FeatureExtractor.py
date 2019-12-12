import glob
import os
import sys
from os import listdir
from os.path import isfile, join

import numpy as np

from BaseInceptionV3Extractor import InceptionV3Extractor
from project_utils import EnsurePath, RescaleList

EXTRACTED_FEATURES_FOLDER='/home/pprusty05/Deep_learning/data/Training/extracted_features_inception_2'


def get_frames_for_sample(source_folder, numbered_folder):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(source_folder, numbered_folder)
    images = sorted(glob.glob(os.path.join(path, '*jpg')))
    return images

def saved_npy(list_of_folders):
    for folders in list_of_folders:
        action_base_name = os.path.basename(folders)
        parent_action_folder_path = os.path.join(EXTRACTED_FEATURES_FOLDER, action_base_name)
        EnsurePath(parent_action_folder_path)
        onlyfolders = [f for f in listdir(folders) if not isfile(join(folders, f))]
        model = InceptionV3Extractor()
        for numbered_folder in onlyfolders:
            images = get_frames_for_sample(source_folder=folders, numbered_folder=numbered_folder)
            path = os.path.join(parent_action_folder_path,str(numbered_folder))
            if os.path.isfile(path + '.npy'):
                continue
            frames = RescaleList(images, 100)
            if frames is not None:
                sequence = []
                for image in frames:
                    features = model.extract(image)
                    sequence.append(features)
                # Save the sequence.
                np.save(path, sequence)


def main():
    list_of_folder_with_extracted_frames = []
    for arg in sys.argv[1:]:
        list_of_folder_with_extracted_frames.append(arg)
    saved_npy(list_of_folder_with_extracted_frames)

if __name__ == "__main__" :
    main()
