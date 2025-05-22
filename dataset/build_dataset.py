import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split

from .TUM import TUM
from .TUM_COND import TUM_COND


def check_dict(key,dict):
    return  True if key in dict else False

def create_labels_dict(**kwargs):
    """
    Creates a dictionary with label column names as keys and provided label values as values.
    *args: Specify the label column names as positional arguments.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    sq_index={'rpm','state','path','label','speed','box'}
    labels_dict = {}

    # Add labels specified as key-value pairs
    for label_name, label_value in kwargs.items():
        if label_name in sq_index:
            labels_dict[label_name] = label_value

    return labels_dict

def split_dataset(datasets_train, target_label=None,isprint=False):
    if target_label is None:
        return datasets_train,datasets_train

    positive_indices = [index for index, label in enumerate(datasets_train.labels) if label == target_label]
    positive_subset = Subset(datasets_train, positive_indices)

    # get subset according to the index
    negative_indices = [index for index, label in enumerate(datasets_train.labels) if label != target_label]
    negative_subset = Subset(datasets_train, negative_indices)

    # print number
    if isprint:
        neg = DataLoader(negative_subset, batch_size=128)
        for i, batch in enumerate(neg):
            print(batch[1])

        pos = DataLoader(positive_subset, batch_size=128)
        for i, batch in enumerate(pos):
            print(batch[1])

    return negative_subset, positive_subset

def get_loaders(train_dataset,
                val_ratio=0.2,
                batch_size=128,
                seed=0,
                with_test=True,
                **kwargs
                ):
    dataset_len = len(train_dataset)
    labels = train_dataset.labels
    ## label same, split random
    if len(set(labels))==1:
        total_size = len(train_dataset)
        train_size = int((1-val_ratio) * total_size)
        valid_size = total_size - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, valid_size])
        return train_dataset,val_dataset

    # StratifiedShuffleSplit
    sss1 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=val_ratio,
                                  random_state=seed)  # for splitting into training and the rest
    sss2 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=0.5,
                                  random_state=seed)  # for splitting the rest into validation and testing

    train_indices, rest_indices = next(sss1.split(range(dataset_len), labels))
    if with_test:

        # Extract the labels of the rest part
        rest_labels = [labels[i] for i in rest_indices]

        # Split the rest part into validation and testing
        val_indices, test_indices = next(sss2.split(rest_indices, rest_labels))

        # Adjust the validation and testing indices to the original indices
        val_indices = [rest_indices[i] for i in val_indices]
        test_indices = [rest_indices[i] for i in test_indices]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        test_subset= Subset(train_dataset,test_indices)
        return train_subset,val_subset,test_subset

    else:
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, rest_indices)
        return train_subset,val_subset

def build_dataset(dataset_type,b=128, normlizetype = '1-1',logger=None,**kwargs):
    # For CW dataset
    # k=[0-9]
    print('build dataset,set kwargs is', kwargs)
    if dataset_type == 'TUM':
        data_dir = "/data/jionkim/signal_dataset/vibtac_res_4000_dir_X/"
        normlizetype = 'none'
        data_set = TUM(data_dir, normlizetype, is_train=True, **kwargs)
        datasets = {'train': data_set}
        cw_data = [i[0] for i in datasets['train']]
        cw_np = np.array(cw_data)
        condition='_cw_ch'+str(data_set.ch)
        return datasets,cw_np,condition
    elif dataset_type == 'TUM_COND':
        data_dir = "/data/jionkim/signal_dataset/npy_part_res_4000_dir_X/"
        normlizetype = 'none'
        data_set = TUM_COND(data_dir, normlizetype, is_train=True, **kwargs)
        datasets = {'train': data_set}
        cw_data = [i[0] for i in datasets['train']]
        cw_np = np.array(cw_data)
        condition='_cw_ch'+str(data_set.ch)
        return datasets,cw_np,condition
    else:
        print("Invalid dataset_type. Choose 'TUM','TUM_COND'")