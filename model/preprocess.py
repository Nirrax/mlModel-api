import json
import os
import librosa
import math

DATASET_PATH = "Data/genres_original"
JSON_PATH = "Data/data.json"

SAMPLE_RATE = 22050
DURATION = 30 # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def saveMfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, number_of_segments=5):

    # dictionary to store data
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    number_of_samples_per_segment = int(SAMPLES_PER_TRACK / number_of_segments)
    expected_number_of_mfcc_vectors_per_segment = math.ceil(number_of_samples_per_segment / hop_length)

    # loop through genres

    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):

        # ensure that we are not in a root folder
        if dir_path is not dataset_path:

            # get label for mapping
            dir_path_components = dir_path.split("\\")
            semantic_label = dir_path_components[-1]
            data["mapping"].append(semantic_label)

            print(f'\nProcessing {semantic_label}')

            # process files
            for file_name in file_names:

                # load audio file
                file_path = os.path.join(dir_path, file_name)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # split file to segments
                for segment in range(number_of_segments):
                    start_sample = number_of_samples_per_segment * segment
                    finish_sample = start_sample + number_of_samples_per_segment

                    # generate mfcc for a segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store value only if is long enough
                    if len(mfcc) == expected_number_of_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print(f'{file_path}, segment: {segment+1}')

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    saveMfcc(DATASET_PATH, JSON_PATH, number_of_segments=10)