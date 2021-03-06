import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import pandas as pd
import torch
import torch.utils.data
import torchaudio
import tqdm
import numpy as np
from transforms import TorchSTFT
import librosa
def load_info(path: str) -> dict:
    """Load audio metadata
    this is a backend_independent wrapper around torchaudio.info
    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds
    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info

def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):
    """Load audio file
    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.
    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)

        return sig, rate
class GunshotForensicDataset(torch.utils.data.Dataset):
    def __init__(self , root: Union[Path, str], seq_duration: Optional[float] = None,target: str = "caliber", source_augmentations: Optional[Callable] = None, resample_freq: int = 44100, n_mels: int =200) -> None:

        self.root = root
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations
        self.df = pd.read_csv('./gun.csv')
        self.target = target
        self.resample_freq = resample_freq
        self.n_mels =  n_mels #hyperparameter we must configure with args
        self.features = list(set(self.df[self.target].to_list()))
        self.features.sort()
        self.num_features = len(self.features)
        
    def get_index(self, classif: str):
        return self.features.index(classif)

    def __getitem__(self, index: int) -> Any:
        start = 0
        path = self.root
        row = self.df.loc[self.df['id'] == index]

        target = row[self.target].to_dict()[index]
        a = row['path']
        
        load_path = "./data" + a.to_dict()[index][1:]
        info = load_info(load_path)
        soundData, sample_rate = load_audio(load_path, dur= self.seq_duration,info=info)
        if self.resample_freq > 0:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_freq)
            soundData = resample_transform(soundData)

        soundData = torch.mean(soundData, dim=0, keepdim=True)

        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.resample_freq, n_mels=self.n_mels)
        melspectrogram = melspectrogram_transform(soundData)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)
        fixed_length = 3 * (self.resample_freq//200)
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(
            melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]
        temp = np.array([self.get_index(target)])
        one_hot = np.zeros((temp.size, self.num_features+1))
        one_hot[np.arange(temp.size), temp] = 1 
        return melspectogram_db, torch.tensor(self.get_index(target),dtype=torch.long)
    def __len__(self) -> int:
        return len(self.df)
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""

if __name__ == "__main__":
    data = GunshotForensicDataset(root="./",seq_duration=1.0)
    mel, target = data[193]
    print(mel)

    print(target)