import sys
import os
import pydrive
import numpy
import json
import pickle
import numpy as np
import pandas as pd

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

total_counts = {'zero':0, 'one':0, 'all':0}

CSV_FOLDER = "home/harsha/hci/tech/OpenFace/build/bin/processed/csv_files/"

for file in os.listdir(CSV_FOLDER):
    print(file)
    df = pd.read_csv(os.path.join(CSV_FOLDER, file))
    ones = df[' success'].sum()
    total_counts['one'] += ones
    total_counts['zero'] += len(df[' success']) - ones
    total_counts['all'] += len(df[' success'])


print(total_counts)
print(total_counts['one'] / total_counts['all'])
