import numpy as np
import pandas as pd
import os

from collections import Counter
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def pre_processing(df):
    x = df[['active_time', 'number_events','number_of_articles_visited']].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[['active_time', 'number_events','number_of_articles_visited']] = x_scaled
    return df

def encode_user(user, trained_enc):
    one_hot_users = trained_enc.transform(user[['os', 'deviceType', 'city', 'country', 'region']])
    one_hot_users = pd.DataFrame(one_hot_users.toarray(), columns=trained_enc.get_feature_names())
    one_hot_users[['active_time', 'number_events','number_of_articles_visited']] = user[['active_time', 'number_events','number_of_articles_visited']].values
    return one_hot_users

def train_encoding(df):
    enc = preprocessing.OneHotEncoder()
    trained_enc = enc.fit(df[['os', 'deviceType', 'city', 'country', 'region']])
    return trained_enc

def one_hot_encoding(df):
    trained_enc = train_encoding(df)

    one_hot_users = encode_user(df, trained_enc)

    pca = PCA(n_components=3)
    pca.fit(one_hot_users)

    one_hot_users_pca = pca.transform(one_hot_users)

    return one_hot_users_pca, trained_enc

def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


class KMeansCB():
    def __init__(self, users, articles, events, batch_size, n_clusters):
        self.users = users
        _, self.trained_enc = one_hot_encoding(users)
        self.articles = articles
        self.events = events
        self.number_of_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.number_of_clusters,
                             init='k-means++',
                             random_state=0)
        self.batch_size = batch_size
        self.trained = False

    def fit(self):
        self.kmeans.fit(encode_user(self.users, self.trained_enc))
        self.trained = True

    def predict(self, user, n_recommendations):
        user_cluster = self.kmeans.predict(encode_user(user, self.trained_enc))
        cluster_users = self.users[self.users.cluster == user_cluster[0]]

        articles_id = self.events[self.events['userId'].isin(cluster_users.user_id)]
        c = Counter(articles_id['documentId'].dropna())
        articles_id = [i[0] for i in c.most_common(n_recommendations)]
        recomended_articles = self.articles[self.articles['document_id'].isin(articles_id)].index
        return recomended_articles

    def evaluate(self, train, test):
        cluster_recommendations = {}
        for cluster in train.cluster.unique():
            cluster_users = train[train.cluster == cluster]
            articles_id = self.events[self.events['userId'].isin(cluster_users.user_id)]
            c = Counter(articles_id['documentId'].dropna())
            top_articles_id = [i[0] for i in c.most_common(self.batch_size)]
            recomended_articles = self.articles[self.articles['document_id'].isin(top_articles_id)].index
            cluster_recommendations[cluster] = recomended_articles
        
        train_eval = []
        test_eval = []
        for _, user in train.iterrows():
            user_cluster = user.cluster
            user_articles_id = self.events[self.events['userId'] == user['user_id']]['documentId'].dropna().unique()
            user_articles = self.articles[self.articles['document_id'].isin(user_articles_id)].index
            
            train_eval.append(len(intersection(user_articles, cluster_recommendations[user_cluster])))

        train_score = np.mean(train_eval) / self.batch_size

        for _, user in test.iterrows():
            user_cluster = user.cluster
            user_articles_id = self.events[self.events['userId'] == user['user_id']]['documentId'].dropna().unique()
            user_articles = self.articles[self.articles['document_id'].isin(user_articles_id)].index
            test_eval.append(len(intersection(user_articles, cluster_recommendations[user_cluster])))

        test_score = np.mean(test_eval) / self.batch_size

        return train_score, test_score

def cb_kmeans_pipeline(user, k, users_df, items_df, events_df):
    # Ignore warnings from sklearn np.matrix
    import warnings
    warnings.filterwarnings("ignore")

    print('Running content-based K-means pipeline ...')

    # Instantiate model
    model = KMeansCB(users=users_df,
                     articles=items_df,
                     events=events_df,
                     batch_size=10,
                     n_clusters=5)

    # Train model
    model.fit()

    # Predict
    recommended_indices = model.predict(user, k).values
    recommendations = items_df[items_df.index.isin(recommended_indices)].values

    # Convert to dataframe
    titles = []
    document_ids = []
    for article in recommendations:
        title = article[1]
        document_id = article[0]
        titles.append(title)
        document_ids.append(document_id)
    rec_df = pd.DataFrame(list(zip(titles, document_ids)), columns =['title', 'documentId'])

    # Evaluate
    train = users_df.sample(frac=0.8)
    test = users_df.drop(train.index)

    train_score, test_score = model.evaluate(train, test)

    return model, rec_df, test_score, train_score



if __name__ == '__main__':
    # Ignore warnings from sklearn np.matrix
    import warnings
    warnings.filterwarnings("ignore")

    # Test, not to be submitted

    # Load data
    users_ = pre_processing(pd.read_csv(PATH + 'users.csv'))
    events_ = pd.read_csv(PATH + 'events.csv')
    articles_ = pd.read_csv(PATH + 'articles.csv')

    # Instantiate model
    model = KMeansCB(users=users_,
                     articles=articles_,
                     events=events_,
                     batch_size=10,
                     n_clusters=5)

    # Train model
    model.fit()

    # Predict
    predictions = model.predict(users_.iloc[[301]], 10)
    print('Predictions:')
    print(predictions)

    # Evaluate
    train = users_.sample(frac=0.75)
    test = users_.drop(train.index)

    train_score, test_score = model.evaluate(train, test)
    print(f'Train score: {train_score}')
    print(f'Test score: {test_score}')
