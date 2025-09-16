import os
import os
import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset
import librosa
from torch.utils.data import DataLoader
import torchvision
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io.wavfile as wav
import torch
import random

def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list

def plot_spk_rec(spk_rec, idx):
    

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    gs = GridSpec(d,d)
    fig= plt.figure(figsize=(30,20),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[idx[i]].T,cmap=plt.cm.gray_r, origin="lower", aspect='auto')
        if i==0:
            plt.xlabel("Time")
            plt.ylabel("Units")    
        
def plot_mem_rec(mem, idx):
    
    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    dim = (d, d)
    
    gs=GridSpec(*dim)
    plt.figure(figsize=(30,20))
    dat = mem[idx]
        
    for i in range(nb_plt):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
        
def get_random_noise(noise_files, size):
    
    noise_idx = np.random.choice(len(noise_files))
    fs, noise_wav = wav.read(noise_files[noise_idx])
    
    offset = np.random.randint(len(noise_wav)-size)
    noise_wav = noise_wav[offset:offset+size].astype(float)
    
    return noise_wav

def generate_random_silence_files(nb_files, noise_files, size, prefix, sr=16000):
    
    for i in range(nb_files):
        
        silence_wav = get_random_noise(noise_files, size)
        wav.write(prefix+"_"+str(i)+".wav", sr, silence_wav)
                
def split_wav(waveform, frame_size, split_hop_length):
    
    splitted_wav = []
    offset = 0
    
    while offset + frame_size < len(waveform):
        splitted_wav.append(waveform[offset:offset+frame_size])
        offset += split_hop_length
        
    return splitted_wav
        
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_root, label_dct, mode, test_id,transform=None, max_nb_per_class=None):
        
        assert mode in ["train", "valid", "test"], 'mode should be "train", "valid" or "test"' 
        
        self.filenames = []
        self.labels = []
        self.mode = mode
        self.transform = transform
        self.test_id = test_id
        
        
        if self.mode == "train" or self.mode == "valid":
            testing_list = txt2list(os.path.join(data_root, "testing_list.txt"))
            validation_list = txt2list(os.path.join(data_root, "validation_list.txt"))
        else:
            testing_list = []
            validation_list = []
        
        
        for root, dirs, files in os.walk(data_root):
            if "_background_noise_" in root:
                continue
            for filename in files:
                if not filename.endswith('.wav'):
                    continue
                command = root.split("/")[-1]
                label = label_dct.get(command)
                if label is None:
                    print("ignored command: %s"%command)
                    break
                partial_path = '/'.join([command, filename])
                
                testing_file = (partial_path in testing_list)
                validation_file = (partial_path in validation_list)
                training_file = not testing_file and not validation_file
                
                if (self.mode == "test") or (self.mode=="train" and training_file) or (self.mode=="valid" and validation_file):
                    if test_id not in filename:
                        full_name = os.path.join(root, filename)
                        self.filenames.append(full_name)
                        self.labels.append(label)
                
        if max_nb_per_class is not None:
            
            selected_idx = []
            for label in np.unique(self.labels):
                label_idx = [i for i,x in enumerate(self.labels) if x==label]
                if len(label_idx) < max_nb_per_class:
                    selected_idx += label_idx
                else:
                    selected_idx += list(np.random.choice(label_idx, max_nb_per_class))
            
            self.filenames = [self.filenames[idx] for idx in selected_idx]
            self.labels = [self.labels[idx] for idx in selected_idx]
        
                
        if self.mode == "train":
            label_weights = 1./np.unique(self.labels, return_counts=True)[1]
            label_weights /=  np.sum(label_weights)
            self.weights = torch.DoubleTensor([label_weights[label] for label in self.labels])
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        item = wav.read(filename)[1].astype(float)
        m = np.max(np.abs(item))
        if m > 0:
            item /= m
        if self.transform is not None:
            item = self.transform(item)
            
        label = self.labels[idx]
        
        return item, label


class SHD_ID(Dataset):

    def __init__(
        self,
        data_root,
        label_dct,
        speaker_id,
        transform=None,
        mode="train",
        max_nb_per_class=None,
        split_ratio=0.8,
        seed=42,
        verbose=True,
    ):
        assert mode in {"train", "test"}, "mode must be 'train' or 'test'"
        self.transform = transform
        self.mode = mode
        self.max_nb_per_class = max_nb_per_class
        rng = random.Random(seed)

        # ── 1. gather speaker files per class ────────────────────────────────
        per_class = {cls: [] for cls in label_dct}  # {class_name: [path, …]}
        for cls in os.listdir(data_root):
            cls_dir = os.path.join(data_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in os.listdir(cls_dir):
                if fn.endswith(".wav") and speaker_id in fn:
                    per_class[cls].append(os.path.join(cls_dir, fn))

        # ── 2. split 80/20, drop under‑k classes, apply k cap ────────────────
        self.filenames = []
        self._class_counts = {}

        for cls, files in per_class.items():
            if not files:
                continue  # speaker never said this word
            rng.shuffle(files)

            cut = int(len(files) * split_ratio)           # size of 80 % slice
            train_files = files[:cut]
            test_files  = files[cut:]

            # ---- drop class if k cannot be satisfied ----
            if (max_nb_per_class is not None) and (len(train_files) < max_nb_per_class):
                continue

            keep = train_files if mode == "train" else test_files

            # ---- enforce k‑shot cap on the *kept* split ----
            if (max_nb_per_class is not None) and (len(keep) > max_nb_per_class):
                keep = keep[:max_nb_per_class]

            self.filenames.extend(keep)
            self._class_counts[cls] = len(keep)

        # cache integer labels for __getitem__
        self.labels = [
            label_dct[os.path.basename(os.path.dirname(p))] for p in self.filenames
        ]

        if verbose:
            self._print_summary()

    # ── torch.Dataset interface ───────────────────────────────────────────────
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        _, wav_data = wav.read(path)          # int16 → float in [‑1, 1]
        wav_data = wav_data.astype(float)
        m = np.max(np.abs(wav_data))
        if m > 0:
            wav_data /= m
        if self.transform is not None:
            wav_data = self.transform(wav_data)
        return wav_data, self.labels[idx]

    # ── helpers ───────────────────────────────────────────────────────────────
    def summary(self):
        """Return {class_name: count} for the current split."""
        return dict(self._class_counts)

    def _print_summary(self):
        k = self.max_nb_per_class if self.max_nb_per_class is not None else "∞"
        total = len(self.filenames)
        speaker = self.filenames[0].split("/")[-1].split("_")[0] if total else "—"
        print(f"\n[SHD_ID] Speaker {speaker} | mode={self.mode} | k={k} | total={total}")
        for cls in sorted(self._class_counts):
            n = self._class_counts[cls]
            print(f"  {cls:<12} {n}")
        print("-" * 36)

class Pad:

    def __init__(self, size):
        
        self.size = size
        
    def __call__(self, wav):
        wav_size = wav.shape[0]
        pad_size = (self.size - wav_size)//2
        padded_wav = np.pad(wav, ((pad_size, self.size-wav_size-pad_size),), 'constant', constant_values=(0, 0))
        return padded_wav
      
class RandomNoise:
    
    
    def __init__(self, noise_files, size, coef):
        
        
        self.size = size
        self.noise_files = noise_files
        self.coef = coef
        
        
    def __call__(self, wav):
        
        
        if np.random.random() < 0.8:
            
            noise_wav = get_random_noise(self.noise_files, self.size)
            noise_power = (noise_wav**2).mean()
            sig_power = (wav**2).mean()
            
            noisy_wav = wav + self.coef  * noise_wav * np.sqrt(sig_power / noise_power) 
            
        else:
            
            noisy_wav = wav
            
        return noisy_wav
     
class RandomShift:
    
    def __init__(self, min_shift, max_shift):
        
        self.min_shift = min_shift
        self.max_shift = max_shift
        
    def __call__(self, wav):
        

        shift = np.random.randint(self.min_shift, self.max_shift+1)
        shifted_wav = np.roll(wav, shift)
    
        if shift > 0:
            shifted_wav[:shift] = 0
        elif shift < 0:
            shifted_wav[shift:] = 0
        
        return shifted_wav
      
class MelSpectrogram:
    
    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order=None, stack=True):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack=stack
        
        
    def __call__(self, wav):
        S = librosa.feature.melspectrogram(
            y=wav,                 # <-- keyword, not positional
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0              # explicit is future-proof
        )

        M = np.max(np.abs(S))
        feat = np.log1p(S / M) if M > 0 else S
    
        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)
        
        elif self.delta_order is not None and self.stack:
            
            feat_list = [feat.T]
            for k in range(1, self.delta_order+1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)
        
        else:
            return np.expand_dims(feat.T, 0)
       
class Rescale:
    
    def __call__(self, input):
        
        std = np.std(input, axis=1, keepdims=True)
        std[std==0]=1
        
        return input/std
    
class WhiteNoise:
    
    def __init__(self, size, coef_max):
        
        
        self.size = size
        self.coef_max = coef_max
        
        
    def __call__(self, wav):
            
        noise_wav = np.random.normal(size = self.size)
        noise_power = (noise_wav**2).mean()
        sig_power = (wav**2).mean()

        coef = np.random.uniform(0., self.coef_max)

        noisy_wav = wav + coef  * noise_wav * np.sqrt(sig_power / noise_power) 
            
        return noisy_wav

def collate_fn(data):
    # stack samples: (B, C, T, F)
    X = torch.tensor([d[0] for d in data])         

    # per-feature standardisation (the line you already had)
    std = X.std(dim=(0, 2), keepdim=True)          # shape (C, 1, F)
    X = X / std

    # ── reshape to (B, T, C*F) ──────────────────────────────────────────
    B, C, T, F = X.shape
    X = X.permute(0, 2, 1, 3).contiguous()         # (B, T, C, F)
    X = X.view(B, T, C * F)                        # (B, T, C*F)

    y = torch.tensor([d[1] for d in data])         # labels

    return X, y

def make_support_loader(shots,
                        root,
                        label_dct, 
                        test_id,
                        transform,
                        batch_size=128,
                        num_workers=2):

    ds = SHD_ID(
        data_root=root,
        label_dct=label_dct,
        mode="train",
        speaker_id=test_id,
        transform=transform,
        max_nb_per_class=shots        
    )
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      collate_fn=collate_fn)

def dataset_init(num_workers = 8,
                 batch_size = 128,
                 time_steps = 100,
                 save_to = "/scratch-node/20235438",
                 shots_list=[1,5,None]):

    sr = 16000
    size = 16000

    n_fft = int(30e-3*sr)
    hop_length = int(10e-3*sr)
    n_mels = 40
    fmax = 4000
    fmin = 20
    delta_order = 2
    stack = True

    melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack) 
    pad = Pad(size)
    rescale = Rescale()

    transform = torchvision.transforms.Compose([pad,
                                 melspec,
                                 rescale])
    test_id = 'c50f55b8'


    data_root = os.path.join(save_to, "gsc_v2_data")
    train_data_root = os.path.join(data_root, "train")
    test_data_root = os.path.join(data_root, "test")
    # valid_data_root = os.path.join(data_root, "val")
    training_words = os.listdir(train_data_root)
    training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x))]
    training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root,x)) if x[0] != "_" ]
    print("{} training words:".format(len(training_words)))
    print(training_words)

    testing_words = os.listdir(test_data_root)
    testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x))]
    testing_words = [x for x in testing_words if os.path.isdir(os.path.join(train_data_root,x)) 
                    if x[0] != "_"]
    print("{} testing words:".format(len(testing_words)))
    print(testing_words)

    testing_words_10 = ['off','stop','yes','left','up','down','right','no','on','go']
    for i in testing_words_10:
        if i not in testing_words: print(i)

    label_dct = {k:i for i,k in enumerate(testing_words + ["_unknown_"])}
    for w in training_words:
        label = label_dct.get(w)
        if label is None:
            label_dct[w] = label_dct["_unknown_"]

    print("label_dct:")
    print(label_dct)

    train_dataset = SpeechCommandsDataset(train_data_root, label_dct, test_id=test_id, transform = transform, mode="train", max_nb_per_class=None)
    train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights,len(train_dataset.weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, sampler=train_sampler, collate_fn=collate_fn)
    # support_dataset = SCD_ID(train_data_root, label_dct, test_id=test_id, transform = transform, mode="train", max_nb_per_class=None)
    # support_dataloader = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    support_dataloaders = {k: make_support_loader(
                            shots=k,
                            root=train_data_root,
                            label_dct=label_dct,
                            test_id=test_id,
                            transform=transform,
                            batch_size=batch_size,
                            num_workers=num_workers)
                       for k in shots_list}
    test_dataset = SpeechCommandsDataset(test_data_root, label_dct, test_id=test_id, transform = transform, mode="test") 
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    query_dataset = SHD_ID(train_data_root, label_dct, speaker_id=test_id, transform = transform, mode="test", max_nb_per_class=None) # Only test data of excluded user
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader, support_dataloaders, query_dataloader


def save_dataloader_to_npy(data_loader, data_file_path='all_data.npy', labels_file_path='all_labels.npy'):
    all_data = []
    all_labels = []

    for data, labels in data_loader:
        all_data.append(data.numpy())
        all_labels.append(labels.numpy())

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_data.shape, all_labels.shape)

    np.save(data_file_path, all_data)
    np.save(labels_file_path, all_labels)
    print(f"Data saved to {data_file_path} and labels saved to {labels_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', type=str, default="/scratch-node/20235438",
                        help='Directory to save generated .npy files and where GSC data is stored')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloaders')
    args = parser.parse_args()

    batch_size = args.batch_size
    save_root = args.save_to

    train, test, support, query = dataset_init(batch_size=batch_size, save_to=save_root)

    os.makedirs(save_root, exist_ok=True)
    data_root = os.path.join(save_root, "gsc_v2_data")

    save_dataloader_to_npy(train,
                           os.path.join(data_root, 'train_data.npy'),
                           os.path.join(data_root, 'train_label.npy'))
    
    save_dataloader_to_npy(test,
                        os.path.join(data_root, 'test_data.npy'),
                        os.path.join(data_root, 'test_label.npy'))

    for k, loader in support.items():
        suffix = "all" if k is None else f"{k}shot"
        save_dataloader_to_npy(loader,
                               os.path.join(data_root, f"support_{suffix}_data.npy"),
                               os.path.join(data_root, f"support_{suffix}_label.npy"))

    save_dataloader_to_npy(query,
                           os.path.join(data_root, 'query_data.npy'),
                           os.path.join(data_root, 'query_label.npy'))

