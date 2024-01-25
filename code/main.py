import os
import numpy as np
from utility.opensmile import process_audio_files
from utility.audio_preprocessing import extract_audio_from_video_files, normalize_loudness_in_audio_files

def __main__():
    datasets = [
        {
            "folder": "../data/VideoTesta/AV",
            "npy_path": "../data/VideoTesta/VideoTestaAV__eGeMAPSv01b_features.npy",
            "csv_path": "../data/VideoTesta/VideoTestaAV__filepaths.csv"
        }
    ]

    for dataset in datasets:
        print(f"Processing {dataset['folder']}...")

        if dataset['folder'] == "../data/VideoTesta/AV":
            extract_audio_from_video_files(workspace=dataset["folder"])
            normalize_loudness_in_audio_files(folder=dataset["folder"])
            folder_to_process = dataset["folder"]
        else:
            raise ValueError(f"Invalid dataset folder: {dataset['folder']}")

        if not os.path.exists(dataset["npy_path"]) and not os.path.exists(dataset["csv_path"]):
            process_audio_files(folder=folder_to_process, npy_path=dataset["npy_path"], csv_path=dataset["csv_path"])
        else:
            print(np.load(dataset["npy_path"]).shape)
            
    print("Done!")

__main__()