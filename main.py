# install the below requirements before running the code
# pip install tensorflow_text
# pip install flair
import json
import pandas as pd
from pandas.io.json import json_normalize
from flair.models import TextClassifier
from flair.data import Sentence
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def main():
    with open("essay-corpus.json", encoding="utf8") as f:
        d = json.load(f)
    df = pd.json_normalize(d, 'claims', ['id', 'confirmation_bias'],
                           record_prefix='claims_')
    idk = df[['id', 'claims_text']]
    pl = df['claims_text'].tolist()
    dz = pd.json_normalize(
        d, 'major_claim', ['id'], record_prefix='major_claim_')
    s = dz.groupby(['id'])['major_claim_text'].apply(';'.join).reset_index()
    k = df.merge(s, on='id')
    w = pd.DataFrame(columns=['confirmation_bias', 'id'])
    w[['confirmation_bias', 'id']] = k[['confirmation_bias', 'id']]
    w.confirmation_bias = w.confirmation_bias.astype('int64')
    k = k[['id', 'major_claim_text', 'claims_text']]

    def normalization(embeds):
        norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
        return embeds / norms
    embed_use = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder/4")
    l = pd.DataFrame(columns=['id', 'emb', 'text'])
    for i, j in k.iterrows():
        maj = j['major_claim_text']
        cla = j['claims_text']
        ids = j['id']
        sent1 = tf.constant([maj])
        sent2 = tf.constant([cla])
        sent1e = embed_use(sent1)
        sent2e = embed_use(sent2)
        sent1e = normalization(sent1e)
        sent2e = normalization(sent2e)
        val = np.matmul(sent1e, np.transpose(sent2e))
        l = l.append({'id': ids, 'emb': val, 'text': cla}, ignore_index=True)

    j = l['emb']
    j = np.float32(j)
    l['change'] = pd.Series(j)
    l = l[['id', 'change', 'text']]
    ls = pd.DataFrame(columns=['senti', 'id'])

    classifier = TextClassifier.load('en-sentiment')
    for index, row in idk.iterrows():
        ids = row['id']
        x = row['claims_text']
        sentence = Sentence(x)
        classifier.predict(sentence)
        sample_str = sentence.labels
        label = sentence.labels[0]
        response = label.value
        ls = ls.append({'senti': response, 'id': ids}, ignore_index=True)

    ls.dtypes
    y = ls['senti']
    y = np.array(list(map(lambda x: 0 if x == "NEGATIVE" else 1, y)))
    ls['sentiment'] = pd.Series(y)
    ls = ls[['id', 'sentiment']]
    result = pd.concat([l, ls], axis=1, join='inner')
    result.columns = ['id', 'change', 'text', 'ids', 'sentiment']
    result = result[['id', 'change', 'text', 'sentiment', ]]
    dk = pd.read_csv(
        r"train-test-split.csv", sep=';')
    dk['ID'] = dk['ID'].str.replace('essay', '')
    tr = dk[dk['SET'] == "TRAIN"]
    te = dk[dk['SET'] == "TEST"]
    tes = te['ID'].tolist()
    tes = list(map(int, tes))
    tra = tr['ID'].tolist()
    tra = list(map(int, tra))
    y_test = w[w['id'].isin(tes)]
    y_testid = y_test['id'].to_numpy()
    y_train = w[w['id'].isin(tra)]
    y_train = y_train[['confirmation_bias']]
    y_test = y_test[['confirmation_bias']]
    x_test = result[result['id'].isin(tes)]
    x_train = result[result['id'].isin(tra)]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(SVC()))])
    parameters = {
        'tfidf__min_df': (0.01, 0.05, 0.1),

    }

    grid_search_tune = GridSearchCV(
        pipeline, parameters, cv=2, n_jobs=2, verbose=3)
    grid_search_tune.fit(x_train['text'], y_train.values.ravel())

    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)
    numeric_feats = ["change"]
    passthrough_feats = ["sentiment", "id"]
    text_feats = "text"
    preproc = make_column_transformer((StandardScaler(), numeric_feats), (CountVectorizer(
        ngram_range=(5, 5)), text_feats),  (TfidfVectorizer(min_df=0.1), text_feats), ("passthrough", passthrough_feats))
    x_train_preproc = preproc.fit_transform(x_train)
    x_test_preproc = preproc.transform(x_test)
    param_grid = {'C': [0.5, 1], 'degree': [1, 2, 3, 4],
                  'gamma': ['auto'], 'kernel': ['rbf', 'sigmoid']}
    grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=2)
    grid.fit(x_train_preproc, y_train.values.ravel())
    print("The best parameters are", grid.best_estimator_)
    grid_predictions = grid.predict(x_test_preproc)
    #print(confusion_matrix(y_test.values.ravel(), grid_predictions))
    # print(classification_report(y_test.values.ravel(), grid_predictions))  # Output
    score = f1_score(y_test.values.ravel(), grid_predictions)
    print("F1-Score: ", score)  # Output
    pred = pd.DataFrame(columns=['id', 'confirmation_bias'])
    pred['confirmation_bias'] = pd.Series(grid_predictions)
    pred['id'] = pd.Series(y_testid)
    pred = pred.replace({0: False, 1: True})
    pred = pred.groupby(['id'], as_index=False).agg(
        {'confirmation_bias': 'first'})
    pred.to_json("predictions.json", orient='records')
    print("it works!")
    pass


if __name__ == '__main__':
    main()
