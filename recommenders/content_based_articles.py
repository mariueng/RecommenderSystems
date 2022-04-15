import numpy as np
import pandas as pd
import os
import re

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer


# Preprocessing helper methods
def make_lower_case(text):
    try:
        return text.lower()
    except:
        return ""

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("norwegian"))
    text = [w for w in text if w not in stops]
    texts = [w for w in text if w.isalpha()]
    texts = " ".join(texts)
    return texts

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def pre_process(df):
    # Retrieve interesting columns
    df = df[['document_id','title','category','keywords', 'author']]
    df.reset_index(inplace=True)

    # Tokenization, lowercase, remove stop words
    df['cleaned_title'] = df['title'].apply(make_lower_case)
    df['cleaned_title'] = df['cleaned_title'].apply(remove_stop_words)
    df['cleaned_title'] = df['cleaned_title'].apply(remove_punctuation)
    df['category'] = df['category'].apply(make_lower_case)

    df['keywords'] = df['keywords'].apply(make_lower_case)
    df['keywords'] = df.keywords.apply(remove_stop_words)
    df['keywords'] = df.keywords.apply(remove_punctuation)

    df['author'] = df['author'].apply(make_lower_case)
    df['author'] = df.author.apply(remove_stop_words)
    df['author'] = df.author.apply(remove_punctuation)

    # We then generate the content text adding the different features
    df['text'] = df.cleaned_title + df.category + df.keywords + df.author

    return df


class CBArticles():
    def __init__(self, df):
        self.df = df
        self.tf = TfidfVectorizer(analyzer='word',
                                  stop_words=set(stopwords.words("norwegian")),
                                  max_df=0.8,
                                  min_df=0.0,
                                  use_idf=True,
                                  ngram_range=(1,3))
        self.tfidf_matrix = self.tf.fit_transform(self.df['text'])

    def predict(self, idxs, no_of_news_article, with_score=False):
        #recomendation vector (sum of watched articles vector)
        recomendation_vector = np.sum(self.tfidf_matrix[idxs],axis=0)
        
        # Get similarity values with other articles
        similarity_score =  enumerate(linear_kernel(self.tfidf_matrix, recomendation_vector))
        
        # Filter visited articles
        similarity_score = [k for k in similarity_score if k[0] not in idxs]
        
        # Sort articles by similarity
        similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the n most similar news articles. Ignore the first movie.
        similarity_score = similarity_score[0:no_of_news_article]
        if with_score:
            return similarity_score

        news_indices = [i[0] for i in similarity_score]

        return news_indices

    def evaluate(self, users, events, batch_size):
        eval_list = []
        for _, user in users.iterrows():
            user_articles_id = events[events['userId'] == user['user_id']]['documentId'].dropna().unique()
            user_articles = self.df[self.df.document_id.isin(user_articles_id)].index

            # Filter articles
            user_articles = [k for k in user_articles if k in self.df.index]

            idxs = np.random.choice(user_articles, batch_size)
            predicted = self.predict(idxs, batch_size)
            intersection = set(user_articles).intersection(predicted)
            eval_list.append(len(intersection))

        return np.mean(eval_list)/batch_size

def cb_articles_pipeline(user, k, users_df, items_df, events_df):
    print('Running content based articles recommender...')

    # Ignore warnings from sklearn np.matrix
    import warnings
    warnings.filterwarnings("ignore")

    # Preprocess data
    df = pre_process(items_df)

    # Instantiate model
    cb = CBArticles(df)

    # Get indices for 10 articles read by user with high activeTime
    user_events = events_df[events_df['userId'] == user]
    user_events_clean = user_events[~user_events['documentId'].isna()].drop_duplicates(subset=['documentId']).sort_values(by=['activeTime'], ascending=False)
    article_ids = user_events_clean.head(10)['documentId'].values

    article_indices = items_df[items_df['document_id'].isin(article_ids)].index.values

    # Predict articles
    recommended_indices = cb.predict(article_indices, k, with_score=True)

    # Convert indices to articles
    recommendations = items_df[items_df.index.isin([x[0] for x in recommended_indices])].values

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
    eval_ = cb.evaluate(users_df, events_df, batch_size=10)

    print('Done!')

    return cb, rec_df, eval_
