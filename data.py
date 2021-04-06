import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import pandas as pd
import torch
import torch.utils.data
import torchaudio
import tqdm

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
    def __init__(self , root: Union[Path, str], seq_duration: Optional[float] = None,target: str = "caliber", source_augmentations: Optional[Callable] = None) -> None:

        self.root = root
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations
        self.df = pd.read_csv('./gun.csv')
        self.target = target
        

    def __getitem__(self, index: int) -> Any:
        start = 0
        path = self.root
        row = self.df.loc[self.df['id'] == index]

        target = row[self.target].to_dict()[index]
        print(target)
        print(row['path'])
        a = row['path']
        
        load_path = "./data" + a.to_dict()[index][1:]
        b = load_audio(load_path)

        return b, target
        
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
    data = GunshotForensicDataset(root="./",seq_duration=5.0)
    print(data[1])
    
