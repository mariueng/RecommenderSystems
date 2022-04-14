import pandas as pd
import os
import json

from collaborative_filtering import cf_pipeline
from content_based_articles import cb_articles_pipeline
from content_based_knn import cb_kmeans_pipeline

PATH = os.getcwd() + '/data/'

def load_active_data(path):
    """
        Load events from files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path, f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if obj is not None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst)

def load_data():
    users = pd.read_csv(PATH + 'users.csv')
    items = pd.read_csv(PATH + 'articles.csv')
    events = pd.read_csv(PATH + 'events.csv')
    events_old = load_active_data(os.getcwd() + '/active1000/')
    return users, items, events, events_old


class HybridRecommender():
    def __init__(self, user, k):
        # self.cf_predict, self.cf_test_eval = cf_pipeline(user, k)
        self.cba_predict, self.cba_test_eval = cb_articles_pipeline(user, k)
        self.cbk_predict, self.cbk_test_eval = cb_kmeans_pipeline(user, k)

    def recommend_articles(self):
        pass


if __name__ == '__main__':
    users, iterms, events, events_old = load_data()
    print('Welcome to this CB-CF Hybrid recommender system')
    print('Please enter the user id:')
    print('If you wish to run with a random user, please enter -1')
    user_id = int(input())
    if user_id < 0:
        user_id = users.sample()['user_id'].values[0]
        print(f'Random user id: {user_id}')
    print('Please enter the number of recommendations:')
    n_recommendations = int(input())
    hybrid = HybridRecommender(user_id, n_recommendations)
    # hybrid.setup()
    # hybrid.recommend_articles(user_id, n_recommendations)
