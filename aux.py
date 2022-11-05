import pandas as pd
import numpy as np
import torch
from utils.nn_data_classifier import load_data, Classifier
from utils.preprocess import preprocess

classified_data_path = './historicals/BTC-min-01-005.csv'
data = pd.read_csv(classified_data_path, parse_dates=['Day']).iloc[:, 1:]
data.sort_values("Day", inplace=True)
# print(data)

SEQ_LEN = 60

# print(data)

processor = preprocess(sequence_length=SEQ_LEN)
training, testing  = processor.preprocess(dataframe=data, validation_size=0.1)

print(training)
print(testing)

#split time series -> split it correctly: validation + main (train/test) -> check
#preprocess: 
# 1.scale (pct_change())
# 2.sequentiate data (seq_len variable)

