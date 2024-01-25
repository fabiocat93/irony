import os
import numpy as np
import pandas as pd
import opensmile

def find_wav_files(root_folder):
    """Find all .wav files in the given folder and its subfolders."""
    wav_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def extract_eGeMAPSv02_features(audio_file):
    """Extract audio features from a given audio file."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(audio_file)
    return features

def process_audio_files(folder, npy_path, csv_path):
    """Process all .wav files in the given folder."""
    audio_files = find_wav_files(folder)
    features = [extract_eGeMAPSv02_features(file) for file in audio_files]
    np.save(npy_path, np.array(features))

    relative_paths = [os.path.relpath(file, folder) for file in audio_files]
    pd.DataFrame(relative_paths, columns=["Relative Path"]).to_csv(csv_path, index=False)
    print(f"Features saved to {npy_path}, paths saved to {csv_path}")
