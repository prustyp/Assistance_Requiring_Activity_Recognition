from pytube import YouTube
import pandas as pd
import os, subprocess
import ffmpeg
import sys
from project_utils import EnsurePath

UNTRIMMED_VIDEOS='/Users/pprusty05/workspace/Deep_learning/data/Training2/untrimmed_videos'
TRIMMED_VIDEOS='/Users/pprusty05/workspace/Deep_learning/data/Training2/trimmed_videos'


def get_filename_without_extension(file_path):
    file_basename = os.path.basename(file_path)
    filename_without_extension = file_basename.split('.')[0]
    return filename_without_extension

def main():
    # initialize an empty list
    list_of_csv =[]
    # arg: set on config(csv file)
    # put all args in the list
    for arg in sys.argv[1:]:
        list_of_csv.append(arg)


    #iterate over the list
    for csv_file in list_of_csv:
        #save the content of csv file in a df
        df = pd.read_csv(csv_file)
        #links = df['youtube_id']
        #file without .cvs extention
        folder_name = get_filename_without_extension(csv_file)
        # The folder where the downloaded videos will be saved
        video_downloaded_path = os.path.join(UNTRIMMED_VIDEOS,folder_name)
        # The folder where the downloaded videos will be trimmed and saved
        trimmed_video_path = os.path.join(TRIMMED_VIDEOS,folder_name)
        # make foloder according to given path if already not exist
        EnsurePath(video_downloaded_path)
        EnsurePath(trimmed_video_path)
        #iterate over the df
        index = 0
        for index, row in df.iterrows():
            try:
                #make the string of youtube link
                link = "https://www.youtube.com/watch?v=" + row['youtube_id']
                # object creation using YouTube which was imported in the beginning
                yt = YouTube(link)
                startTime = row['time_start']
                endTime = row['time_end']
            except Exception as e:
                print(str(e)) # to handle exception
            stream = yt.streams.filter(file_extension='mp4').first()
            noError = True
            try:
                # downloading the video and save as index(0.mp4,1.mp4)
                fname_index = str(index)
                downloaded_path = stream.download(output_path=video_downloaded_path, filename=fname_index)
            except Exception as e:
                print(str(e))
                noError = False
            finally:
                if noError:
                    #If there is noError then only go for trimming the video
                    file_name = os.path.basename(downloaded_path)
                    target_path = os.path.join(trimmed_video_path,file_name)
                    command_name = "ffmpeg -i " + downloaded_path + " -ss  " + str(startTime) + " -to " + str(endTime) + " -c copy " + target_path
                    print(command_name)
                    os.system(command_name)
                    index = index + 1


if __name__ == "__main__" :
    main()

