import pandas as pd
import numpy as np
import torch
from utils.nn_data_classifier import load_data, Classifier

classified_data_path = './historicals/BTC-min-01-005.csv'
data = pd.read_csv(classified_data_path)
print(data)
