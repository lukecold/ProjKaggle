import numpy as np
from math import *

class RandomForest:
    SURVIVED = "Survived"
    trees = []
    
    def grow(data):
        features = data.columns.drop(SURVIVED)
        column_num = features.size
        column_sample_num = floor(sqrt(column_num))
        column_sample = np.random.choice(column_num, column_sample_num, replace = False)

        # find the significant column and prediction TODO: deal with the non-numerial columns
        significant_column = None
        best_se = sys.maxsize
        best_split_point = None
        for column_idx in column_sample:
            column_name = features[column_idx]
            split_point, se = find_split_point(data, column_name)
            if (se < best_se):
                best_se = se
                best_split_point = split_point
                significant_column = column_name

        trees.append(lambda test_data: {
            left = test_data[test_data[significant_column] < best_split_point][SURVIVED]
            right = test_data[test_data[significant_column] >= best_split_point][SURVIVED]
            round(left.mean(axis=0)), round(right.mean(axis=0))
        })

    def predict(data):
        score = 0
        for predict in trees:
            score += predict(data)
        return round(score / trees.size)

    def find_split_point(data, column_name):
        best_split_point = 0
        best_se = sys.maxsize
        for split_point in set(data[column_name]):
            se = split(data, column_name, split_point)
            if (se < best_se):
                se = best_se
                best_split_point = split_point
        return best_split_point, best_se


    def split(data, column_name, split_point):
        left, right = data[data[column_name] < split_point][SURVIVED], data[data[column_name] >= split_point][SURVIVED]
        left_predict, right_predict = round(left.mean(axis=0)), round(right.mean(axis=0))
        se = 0
        for l in left:
            se += pow(l - left_predict, 2)
        for r in right:
            se += pow(r - right_predict, 2)
        return se