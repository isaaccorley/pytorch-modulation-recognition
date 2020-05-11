import os
import gc
import itertools
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict


# Modulation types
MODULATIONS = {
    "QPSK": 0,
    "8PSK": 1,
    "AM-DSB": 2,
    "QAM16": 3,
    "GFSK": 4,
    "QAM64": 5,
    "PAM4": 6,
    "CPFSK": 7,
    "BPSK": 8,
    "WBFM": 9,
}

# Signal-to-Noise Ratios
SNRS = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2
]


class RadioML2016(torch.utils.data.Dataset):

    URL = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2"
    modulations = MODULATIONS
    snrs = SNRS

    def __init__(
        self,
        data_dir: str = "./data",
        file_name: str = "RML2016.10b.dat"
    ):
        self.file_name = file_name
        self.data_dir = data_dir
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Load data from file """
        print("Loading dataset from file...")
        with open(os.path.join(self.data_dir, self.file_name), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        X, y = [], []
        print("Processing dataset")
        for mod, snr in tqdm(list(itertools.product(self.modulations, self.snrs))):
            X.append(data[(mod, snr)])

            for i in range(data[(mod, snr)].shape[0]):
                y.append((mod, snr))

        X = np.vstack(X)

        return X, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Load a batch of input and labels """
        x, (mod, snr) = self.X[idx], self.y[idx]
        y = self.modulations[mod]
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]

    def get_signals(self, mod: List[str] = None, snr: List[int] = None) -> Dict:
        """ Return signals of a certain modulation or signal-to-noise ratio """

        # If None then it means all mods or snrs
        if mod is None:
            modulations = self.modulations.copy()
        if snr is None:
            snrs = self.snrs.copy()

        # If single mod or snr then convert to list to make iterable
        if not isinstance(mod, List):
            modulations = [mod]
        if not isinstance(snr, List):
            snrs = [snr]

        # Aggregate signals into a dictionary
        X = {}
        for mod, snr in list(itertools.product(modulations, snrs)):
            X[(mod, snr)] = []
            for idx, (m, s) in enumerate(self.y):
                if m == mod and s == snr:
                    X[(mod, snr)].append(np.expand_dims(self.X[idx, ...], axis=0))
            
            X[(mod, snr)] =  np.concatenate(X[(mod, snr)], axis=0)

        return X