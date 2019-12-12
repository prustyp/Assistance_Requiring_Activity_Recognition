import os
import pathlib

def EnsurePath(full_path):
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


def RescaleList(input_list, size):
    '''
    Select specific frames from input_list depending on the size
    :param input_list:
    :param size:
    :return:
    '''
    if (len(input_list) < size):
        return None

    # Get the number to skip between iterations.
    skip = len(input_list) // size

    #Build the new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    #Remove last one if needed.
    return output[:size]
