from os import listdir
from os.path import isfile, join
import os
import cv2
import sys

from project_utils import EnsurePath


EXTRACTED_FRAMES_FOLDER='/Users/pprusty05/workspace/Deep_learning/data/Training2/extracted_frames'

def get_filename_without_extension(file_path):
    file_basename = os.path.basename(file_path)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension

def main():
    #initialize an empty list
    list_of_folder_with_videos = []
    #arg: set on config(csv file)
    #put all args in the list
    for arg in sys.argv[1:]:
        list_of_folder_with_videos.append(arg)

    #Extract frames
    for folders in list_of_folder_with_videos:
        main_folder_name = os.path.basename(folders)
        frames_extracted_main_folder = os.path.join(EXTRACTED_FRAMES_FOLDER, main_folder_name)
        EnsurePath(frames_extracted_main_folder)
        onlyfiles = [f for f in listdir(folders) if isfile(join(folders, f))]
        for file_name in onlyfiles:
            folder_base_name = os.path.splitext(file_name)[0]
            new_folder_path = os.path.join(frames_extracted_main_folder, folder_base_name)
            EnsurePath(new_folder_path)
            source_file_path = os.path.join(folders,file_name)
            vidcap = cv2.VideoCapture(source_file_path)

            # #Create the directory into for the file
            video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            print("Number of frames: ", video_length)
            count = 0
            print("Converting video..\n")
            # Start converting the video
            while vidcap.isOpened():
                # Extract the frame
                ret, frame = vidcap.read()
                # Write the results back to output location.
                cv2.imwrite(new_folder_path + "/%#05d.jpg" % (count + 1), frame)
                count = count + 1
                # If there are no more frames left
                if (count > (video_length - 1)):
                    # Log the time again

                    # Release the feed
                    vidcap.release()
                    # Print stats
                    print("Done extracting frames.\n%d frames extracted" % count)
                    #print("It took %d seconds forconversion." % (time_end - time_start))
                    break


if __name__ == "__main__" :
    main()