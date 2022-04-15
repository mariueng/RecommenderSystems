from collaborative_filtering import cf_pipeline
from content_based_articles import cb_articles_pipeline
from content_based_kmeans import cb_kmeans_pipeline


class HybridRecommender():
    def recommend_articles(self, user, k, events, users_df, items_df):
         # Check whether user exists in history
        if user not in events['userId'].values:
            _, rec, _ = cb_articles_pipeline(user, k, users_df, items_df, events)
            return rec
        # Make recommendations
        _, rec, scores = cf_pipeline(user, k, events)
        if scores['mse'] < 0.005:
            return rec

        _, rec, scores = cb_kmeans_pipeline(users_df[users_df['user_id'] == user], k, users_df, items_df, events)
        return rec
