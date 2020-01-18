import os
import sys
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset, Subset

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants


from torch_datasets import PoseDataset

data = PoseDataset(3, 5, False, False, None)
train_indices, dev_indices = data.split_data(0)
dev_data = Subset(data, train_indices)
labels = np.array([item['labels'] for item in dev_data])
print(len(labels))
counts = {}
counts[0] = np.sum(labels[:,0] == 1)
counts[1] = np.sum(labels[:,1] == 1)
counts[2] = np.sum(labels[:,2] == 1)
counts[3] = np.sum(labels[:,3] == 1)
print(counts)
