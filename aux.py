import pandas as pd
import numpy as np
import torch
from utils.nn_data_classifier import load_data, Classifier

classified_data_path = './historicals/BTC-min-01-005.csv'
data = pd.read_csv(classified_data_path)
print(data)

#split time series -> split it correctly: validation + main (train/test) -> check
#preprocess: 
# 1.scale (pct_change())
# 2.sequentiate data (seq_len variable)

