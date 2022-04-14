import os
import json
import pandas as pd
import numpy as np


def load_events(path):
    """
        Load events from original files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if obj is not None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst)


def load_csv(path):
    """
    Load csv files
    
    Args:
        - path: path to csv file
    Returns:
        - df: pandas dataframe
    """

    df = pd.read_csv(path)
    df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    df.rename(columns={'id': 'documentId'}, inplace=True)
    return df


def train_test_split(ratings, fraction=0.2):
    """Leave out a fraction of dataset for test use"""
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        size = int(len(ratings[user, :].nonzero()[0]) * fraction)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=size, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    return train, test


if __name__ == "__main__":
    load_csv('/Users/mariu/dev/school/TDT4215/tdt4215-2022/data/all_data.csv')
