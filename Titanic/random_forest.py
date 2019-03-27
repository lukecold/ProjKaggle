import numpy as np
from math import *

class RandomForest:
    trees = []
    
    def grow(data):
        features = data.columns.drop("Survived")
        column_num = features.size
        column_sample_num = floor(sqrt(column_num))
        column_sample = np.random.choice(column_num, column_sample_num, replace = False)

        # find the significant column and prediction
        significant_column
        column_name = features[significant_column]
        predict = lambda value: value > 1

        trees.append(lambda test_data: predict(test_data[column_name]))

    def predict(data):
        score = 0
        for predict in trees:
            score += predict(data)
        return round(score / trees.size)