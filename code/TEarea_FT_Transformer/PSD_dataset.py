from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from PSD_ExtractFeatures import *

def random_blank(eeg, num_range=10, long_range=10):
    num = np.random.randint(0, num_range)
    starts = np.random.randint(0, 430, num)
    longs = np.random.randint(1, long_range, num)
    ends = starts + longs
    ends = [440 if i > 440 else i for i in ends]
    for s, e in zip(starts, ends):
        eeg[s:e] = 0
    return eeg

class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path, opt):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]
        # Compute size
        self.size = len(self.data)
        self.opt = opt

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = ((self.data[i]["eeg"].float() - self.means) / self.stddevs)  # .t() # CxT
        # ================取TE区域的电极====================
        # 左侧：F9(66),FT9(41),TP9(17),P9(117)
        # 右侧：F10(71),FT10(46),TP10(22),P10(124)
        channels = [65, 40, 16, 116, 70, 45, 21, 123]
        # channels = [65, 1, 2, 3, 4, 5, 6, 7]
        # TE_eeg = eeg[65, :].unsqueeze(0)
        # for c in range(1, 8):
        #     TE_eeg = torch.cat((TE_eeg, eeg[channels[c], :].unsqueeze(0)), dim=0)
        # time_eeg = TE_eeg
        # fre_eeg = TE_eeg
        # ================去除TE区域
        # Delete_TE_eeg = eeg[0, :].unsqueeze(0)
        # for c in range(1, eeg.shape[0]):
        #     if(c not in channels):
        #         Delete_TE_eeg = torch.cat((Delete_TE_eeg, eeg[c, :].unsqueeze(0)), dim=0)
        # TE_eeg = Delete_TE_eeg
        # time_eeg = Delete_TE_eeg
        # fre_eeg = Delete_TE_eeg
        # ================去除TE区域
        time_eeg = eeg
        fre_eeg = eeg
        # Check filtering
        # Uses global opt
        if self.opt.filter_low or self.opt.filter_notch or self.opt.filter_high or self.opt.filter_notch_harmonics or self.opt.leave_notch or self.opt.leave_notch_high or self.opt.leave_10:
            # preprocess time eeg Time axis
            N = eeg.size(1)
            T = 1.0 / 1000.0
            time = np.linspace(0.0, N * T, N)
            # Frequency axis2
            w = rfftfreq(N, T)
            # FFT
            eeg = eeg.numpy()
            eeg_fft = rfft(eeg)
            # Filter
            eeg_fft[:, w < 7] = 0
            # eeg_fft[:, np.bitwise_and(w > 47, w < 53)] = 0
            eeg_fft[:, w > 71] = 0
            time_eeg = irfft(eeg_fft)
            # Convert to tensor
            time_eeg = torch.tensor(time_eeg)
            # preprocess frequency eeg
            fre_eeg = PSD_Etract(eeg)
            fre_eeg = torch.tensor(fre_eeg)
            # fre_eeg = fre_eeg.unsqueeze(1)
        # Transpose to TxC
        time_eeg = time_eeg.t()
        time_eeg = time_eeg[20:460, :]
        # Get label
        label = self.data[i]["label"]
        # Return
        return fre_eeg, time_eeg, label


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)
        self.split_name = split_name

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        fre_eeg, time_eeg, label = self.dataset[self.split_idx[i]]
        if self.split_name == 'train':
            fre_eeg = fre_eeg.numpy()
            # eeg = random_flip(eeg)
            fre_eeg = random_blank(fre_eeg)
            time_eeg = time_eeg.numpy()
            time_eeg = random_blank(time_eeg)
        # Return
        return fre_eeg, time_eeg, label
