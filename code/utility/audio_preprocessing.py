import ffmpeg
import os
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
import pyloudnorm as pyln

def extract_audio_from_video_files(workspace):
    """Extract audio from all .mp4 files in the given folder."""
    video_files = find_mp4_files(workspace)
    for file in video_files:
        extract_audio(file)

def find_mp4_files(folder):
    """Find all .mp4 files in the given folder and its subfolders."""
    mp4_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def extract_audio(video_file):
    """Extracts the audio from the given video file."""
    if not os.path.exists(f'{video_file[:-4]}.wav'):
        (
            ffmpeg
            .input(video_file)
            .output(f'{video_file[:-4]}.wav', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .run()
        )

# find all .wav files in the given folder
def find_wav_files(folder):
    """Find all .wav files in the given folder and its subfolders."""
    wav_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def normalize_loudness_in_audio_files(folder, new_loudness=-23.0):
    audio_files = find_wav_files(folder)
    for file in audio_files:
        data, rate = sf.read(file) # load audio
        meter = pyln.Meter(rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(data) # measure loudness
        if loudness != new_loudness:
            data = pyln.normalize.loudness(data, loudness, new_loudness) # loudness normalize audio to new_loudness dB LUFS
            sf.write(file, data, rate)