import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
from utils.sequence_transform import *

signal_size = 1024
from collections import Counter

# label
label_all = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9

# generate Training Dataset and Testing Dataset
def get_files(root):
    data, lab = data_load(root, label=0)
    return [data, lab]

def data_load(root, label):

    file_list = glob.glob(root + '/*.npy')
    data = []
    for file_name in file_list:
        data_p = np.load(file_name)
        data_p = np.expand_dims(data_p, axis=1)
        data.append(data_p)
    lab = [label] * len(data)

    return data, lab


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            # Reshape(),
            Normalize(normlize_type),
            # RandomAddGaussian(),
            # RandomScale(),
            # RandomStretch(),
            # RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            # Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]

class TUM(Dataset):

    def __init__(self, data_dir, normlizetype, is_train=True,ch=None,**kwargs):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.ch=self._get_ch(ch)

        list_data = get_files(self.data_dir)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = self.data_pd['label'].tolist()
            self.transform = None
        self.cls_num = set(list_data[1])

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        if self.is_train:
            data = self.seq_data[idx]
            label = self.labels[idx]

            if self.transform:
                data = self.transform(data)
            return data, label
        else:
            data = self.seq_data[idx]
            label = self.labels[idx]
            if self.transform:
                data = self.transform(data)
            return data,label

    def get_classes_num(self):
        return len(self.cls_num),self.cls_num# num, name

    def _get_ch(self,ch):
        if ch is not None:
            return ch
        else :
            raise('CW data ch is None')

def counter_dataloader(data_loader):

    label_counts = Counter()
    for batch in data_loader:
        _, labels = batch
        for label in labels:
            label_counts[label.item()] += 1
    print('dataloader counter',label_counts)

def get_loaders(train_dataset, seed, batch, val_ratio=0.2):
    dataset_len = len(train_dataset)
    labels = train_dataset.labels

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_indices, val_indices = next(sss.split(range(dataset_len), labels))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)
    counter_dataloader(val_dataloader)
    return train_dataloader, val_dataloader