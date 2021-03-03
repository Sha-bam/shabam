import struct
import os
import stempeg
import librosa
import numpy as np
import librosa.display
'''
Pulled from: https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7
'''


class WavFileHelper():

    def read_file_properties(self, filename):

        wave_file = open(filename, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)


def load_data(path):
    for folder in os.listdir(path):
        a = folder.split('_')
        gun_name = a[0]
        recording_method = a[1]
        print(gun_name, recording_method)
        for file in os.listdir(f'{path}/{folder}'):
            print(file)
            if file[0] != ".":
                file_path = f'{path}/{folder}/{file}'
                print(file_path)
                S, rate = stempeg.read_stems(file_path)


def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled
