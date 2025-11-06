import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader, random_split

class SpikeData(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels, stage):
        self.root_dir = root_dir
        self.stage = stage
        self.data_list = os.path.join(root_dir, stage)
        self.data_list = sorted(os.listdir(self.data_list))
        self.labels = labels
        self.length = len(self.data_list)
    
    def __getitem__(self, idx: int):
        data = np.load(os.path.join(self.root_dir,self.stage,self.data_list[idx]))
        spk = data['spk'].astype(np.float32)
        pixel_offset = 13
        spk = spk[:, pixel_offset:250-pixel_offset, pixel_offset:250-pixel_offset] # [200,250,250] -> [200,224,224]
        label_idx = int(data['label'])
        label = self.labels[label_idx]
        return spk, label, label_idx

    def __len__(self):
        return self.length

def get_loader(root_dir, labels, split="train", batch_size=4, num_workers=None, val_split_ratio=0.2, seed=None, device=None, pin_memory=None):
    """
    Get data loader for train, validation, or test split.
    
    Args:
        root_dir: Root directory containing train/ and test/ subdirectories
        labels: List of class labels
        split: One of 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers (None = auto-detect based on device)
        val_split_ratio: Fraction of training set to use for validation (default: 0.2)
        seed: Random seed for train/val split (for reproducibility)
        device: PyTorch device (torch.device) - used to set pin_memory and num_workers
        pin_memory: Pin memory flag (None = auto-detect based on device)
    """
    # Device-aware settings
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Set pin_memory for GPU (CUDA only, not MPS)
    if pin_memory is None:
        pin_memory = device.type == 'cuda'
    
    # Auto-set num_workers if not specified
    if num_workers is None:
        if device.type == 'cuda':
            num_workers = 4  # More workers for GPU
        else:
            num_workers = 2  # Fewer workers for CPU
    
    if split == 'test':
        # Test set is separate
        dataset = SpikeData(root_dir, labels, 'test')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
        return loader
    
    elif split in ['train', 'val']:
        # Load full training set
        full_train_dataset = SpikeData(root_dir, labels, 'train')
        
        # Split into train and validation
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        total_size = len(full_train_dataset)
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        if split == 'train':
            dataset = train_dataset
            shuffle = True
        else:  # split == 'val'
            dataset = val_dataset
            shuffle = False
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
        return loader
    
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../data', 'u-caltech')
    labels = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
    dataset = SpikeData(data_dir, labels, 'train')
    spk,label, label_idx = dataset[random.randint(0, len(dataset) - 1)]
    print(spk.shape, label, label_idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    spikes = batch[0]  # spk
    labels = batch[2]  # label_idx
    print(spikes.shape, labels.shape)
    