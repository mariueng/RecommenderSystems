"""

Code inspired by example project and original author Ethan Rosenthal
Link to original article: https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/

"""

import pandas as pd
import numpy as np

from numpy.linalg import solve
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


def pre_processing(df):
    # Impute missing activeTime values by the mean
    df['activeTime'].fillna(df['activeTime'].mean(), inplace=True)

    # Normalize activeTime values in a new column
    scaler = preprocessing.MinMaxScaler()
    df['normalized_activeTime'] = scaler.fit_transform(df['activeTime'].values.reshape(-1,1))

    return df

def prepare_data(df):
    # Drop all documents with no ID
    df = df[~df['documentId'].isnull()]
    # Drop all duplicates of same user and document
    df = df.drop_duplicates(subset=['userId', 'documentId']).reset_index(drop=True)
    # Sort by userId and time
    df = df.sort_values(by=['userId', 'time'])
    n_users = df['userId'].nunique()
    n_items = df['documentId'].nunique()

    # Intialize ratings matrix
    ratings = np.zeros((n_users, n_items))
    # Check that there are no duplicate users
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    # Convert to truth matrix
    new_user = np.r_[True, new_user]
    # Create new user ID: uid
    df['uid'] = np.cumsum(new_user)
    # Get all unique documents
    item_ids = df['documentId'].unique().tolist()
    # Create new dataframe with document and new document ID: tid
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    # Merge uid and tid on documentId
    df = pd.merge(df, new_df, on='documentId', how='outer')
    # Create our lookup table for interactions between users and documents
    df_ext = df[['uid', 'tid', 'normalized_activeTime']]
    # Return lookup table, empty ratings matrix and mapper to lookup documentId
    mapper = df
    return df_ext, ratings, mapper

def fill_ratings_matrix(df_ext, ratings):
    # Fill ratings matrix with normalized activeTime as implicit feedback
    for row in df_ext.itertuples():
        ratings[row[1] - 1, row[2] - 1] = row[3]
    return ratings

def train_test_split(ratings, fraction):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        size = int(len(ratings[user, :].nonzero()[0]) * fraction)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=size,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    
    assert(np.all((train * test) == 0))
    return train, test


class CFModel:
    def __init__(self,
                 ratings,
                 n_factors=5,
                 item_reg=0.01,
                 user_reg=0.01,
                 verbose=False):
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose

    def als_step(self, latent_vectors, fixed_vecs,
                 ratings, _lambda, type='user'):
        """
        Alternating Least Squares for training process.
        
        Params:
            latent_vectors: (2D array) vectors need to be adjusted.
            fixed_vecs: (2D array) vectors fixed.
            ratings: (2D array) rating matrx.
            _lambda: (float) regularization coefficient. 
        """
        if type == 'user':
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambda_i = np.eye(YTY.shape[0]) * _lambda
            
            for u in range(latent_vectors.shape[0]):
                latent_vectors[u,:] = solve((YTY + lambda_i),
                              ratings[u,:].dot(fixed_vecs))
        elif type == 'item':
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambda_i = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i,:] = solve((XTX + lambda_i),
                              ratings[:,i].T.dot(fixed_vecs))
                
        return latent_vectors
    
    def train(self, n_iter=10):
        # initialize latent vectors for training process
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))
        
        self.partial_train(n_iter)

    def partial_train(self, n_iter):
        """train model for n_iter iterations."""
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print("Current iteration: {}".format(ctr))
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs,
                                           self.ratings, self.user_reg,
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs,
                                           self.ratings, self.item_reg,
                                           type='item')
            
            ctr += 1

    def predict_all(self):
        """Predict ratings"""
        predictions = np.zeros((self.user_vecs.shape[0],
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u,i] = self.user_vecs[u,:].dot(self.item_vecs[i, :].T)
        
        return predictions

    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    
    def evaluate(self, pred, actual):
        """Evaluate recommendations according to recall@k and ARHR@k"""
        total_num = len(actual)
        tp = 0.
        arhr = 0.
        for pred_val in pred:
            if pred_val in actual:
                tp += 1.
                arhr += 1./(np.argwhere(actual == pred_val).flatten()[0] + 1.)
        recall = tp / float(total_num)
        arhr = arhr / len(actual)

        return recall, arhr

    def get_mse(self, pred, actual):
        """Calculate mean squard error between actual ratings and predictions"""
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE during train and test iterations.
        
        Params:
            iter_array: (list) List of numbers of iterations to train for each step of 
                        the learning curve.
            test: (2D array) test dataset.
        """
        iter_array.sort()
        self.train_mse = []
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print("Iteration: {}".format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)
            
            predictions = self.predict_all()
            
            self.train_mse += [self.get_mse(predictions, self.ratings)]
            self.test_mse += [self.get_mse(predictions, test)]
            if self._v:
                print("Train mse: {}".format(str(self.train_mse[-1])))
                print("Test mse: {}".format(str(self.test_mse[-1])))
            iter_diff = n_iter
        
        return predictions


# Methods to retrieve recommended articles for a single user
def user_cosine_similarity(model):
    sim = model.user_vecs.dot(model.user_vecs.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim / norms / norms.T

def retrieve_top_k_articles(user, similarity, mapper, df_ext, k, model):
    # Indexes of cosine similarity matrix matches user ids
    if isinstance(user, int):
        # uid input, use direct lookup
        user_idx = df_ext['uid'][user]
    elif isinstance(user, str):
        # user identifier input, lookup uid
        user_idx = mapper[mapper['userId'] == user]['uid'].values[0]
    else:
        raise ValueError("user must be int (0, 1000) or str identifier")

    # Get top 10 similar users
    top_10_similar_user_ids = np.argsort(similarity[user_idx, :])[::-1]

    top_10_article_ids = []
    # Do not start on 1, same user as test user
    i = 1
    for i in range(1, k + 1):
        # Get the index in the rating matrix for the item in question
        user_id = top_10_similar_user_ids[i]
        article_id = np.argmax(model.ratings[user_id])
        top_10_article_ids.append(article_id)

    recommended_articles = []
    # Translate to title using df_ext
    for tid in top_10_article_ids:
        article = mapper[mapper['tid'] == tid].head(1)
        recommended_articles.append(article.values[0])

    return recommended_articles

def cf_pipeline(user, k, events_df):
    print('Running collaborative filtering pipeline ...')

    # Pre-process data
    events = pre_processing(events_df)

    """
    Optimal params for Implicit Collaborative Filtering using Matrix Factorization:

    'n_factors': 5,
    'reg': 0.01,
    'n_iter': 100,
    """

    # Prepare data and create ratings matrix
    df_ext, ratings, mapper = prepare_data(events)
    ratings_active_time = fill_ratings_matrix(df_ext, ratings)

    # Split data into train and test
    train, test = train_test_split(ratings_active_time, 0.2)

    # Instantiate model
    model = CFModel(train, verbose=False)

    # Train model
    iter_array = [1, 2, 5, 10, 25, 50, 100]
    model.calculate_learning_curve(iter_array, test)

    # Display top 10 articles for a user
    als_user_sim = user_cosine_similarity(model)

    recommendations = retrieve_top_k_articles(user, als_user_sim, mapper, df_ext, k, model)
    titles = []
    document_ids = []
    for article in recommendations:
        title = article[3]
        document_id = article[8]
        titles.append(title)
        document_ids.append(document_id)
    rec_df = pd.DataFrame(list(zip(titles, document_ids)), columns =['title', 'documentId'])

    # Evaluation
    # Is done between predicted and actual articles from the test set
    # Find true article ids (tid) the user read based on test set
    ## Get tid from test matrix argmax and use events to match with userId, if the article exists
    tids = np.unravel_index(np.argsort(test.ravel())[-10:], test.shape)[1]
    # Retrieve true articles from mapper
    test_articles = mapper[(mapper['userId'] == user) & (mapper['tid'].isin(tids))]['documentId'].values

    evaluate = model.evaluate(pred=document_ids, actual=test_articles.tolist())

    scores = {'mse': model.test_mse[-1], 'recall': evaluate[0], 'arhr': evaluate[1]}

    print('Done!')

    return model, rec_df, scores
