import numpy as np
import pandas as pd
import random_forest as rf
from math import *

sample_num = 1500

train_data = pd.read_csv("~/workspace/ProjKaggle/Titanic/train.csv")
row_num = train_data.index.size
sample_size = floor(sqrt(row_num))

predictor = rf.RandomForest

for i in range(sample_num):
    predictor.grow(train_data.sample(sample_size))

test_data = pd.read_csv("~/workspace/ProjKaggle/Titanic/test.csv")
predictor.predic(test_data)