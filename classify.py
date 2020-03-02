import pandas as pd
import numpy as np
import nltk
w = nltk.word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import nltk
from nltk.corpus import stopwords

coms = pd.read_csv('~/Desktop/text-model/Coms.csv')

def clean_text(df, column):
    coms['Text_clean'] = coms['Comment'].str.replace(r'[^A-Za-z0-9]','')
    coms['Text_clean'] = coms['Comment'].str.lower()
    return df

coms_cl = clean_text(coms, 'Comment')
# print(coms['Labels'].value_counts())
# print(coms_cl.head())

coms['Token'] = coms['Comment'].apply(w)

x = coms_cl['Comment'].tolist()
y = coms_cl['Class_Label'].tolist()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
bow = CountVectorizer(stop_words={'english'})
x_train_c = bow.fit_transform(x_train)
x_test_c = bow.transform(x_test)
nb = MultinomialNB()
y_pred_nb = nb.fit(x_train_c, y_train).predict(x_test_c)
accuracy_score(y_test, y_pred_nb)
print(accuracy_score(y_test, y_pred_nb))

tf1 = TfidfVectorizer(stop_words={'english'})
x_train_tf = tf1.fit_transform(x_train)
x_test_tf = tf1.transform(x_test)
y_pred_nb_tf = nb.fit(x_train_tf, y_train).predict(x_test_tf)
print(accuracy_score(y_test, y_pred_nb_tf))
# print(coms['Token'])

#################################################################

bow2 = CountVectorizer()
dtm = bow2.fit_transform(coms['Comment'])
#print(bow2.get_feature_names())
# print(dtm.toarray())

tf = TfidfTransformer()
dtm1 = tf.fit_transform(dtm.toarray())
# print(dtm1.toarray())

tf1 = TfidfVectorizer()
dtm2 = tf1.fit_transform(coms['Comment'])
dtm2.toarray()

# from sklearn.feature_extraction.text import CountVectorizer
# import re
# stop_words = set(stopwords.words("english"))
# cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,1))
#
# X=cv.fit_transform(coms['Comment'])
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(X)
# # get feature names
# feature_names=cv.get_feature_names()
#
# # fetch document for which keywords needs to be extracted
# doc=coms['Comment']
#
# #generate tf-idf for the given document
# tf_idf_vector=dtm2
#
# #Function for sorting tf_idf in descending order
# from scipy.sparse import coo_matrix
# def sort_coo(coo_matrix):
#     tuples = zip(coo_matrix.col, coo_matrix.data)
#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
#
# def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#     """get the feature names and tf-idf score of top n items"""
#
#     #use only topn items from vector
#     sorted_items = sorted_items[:topn]
#
#     score_vals = []
#     feature_vals = []
#
#     # word index and corresponding tf-idf score
#     for idx, score in sorted_items:
#
#         #keep track of feature name and its corresponding score
#         score_vals.append(round(score, 3))
#         feature_vals.append(feature_names[idx])
#
#     #create a tuples of feature,score
#     #results = zip(feature_vals,score_vals)
#     results= {}
#     for idx in range(len(feature_vals)):
#         results[feature_vals[idx]]=score_vals[idx]
#
#     return results
# #sort the tf-idf vectors by descending order of scores
# sorted_items=sort_coo(tf_idf_vector.tocoo())
# #extract only the top n; n here is 10
# keywords=extract_topn_from_vector(feature_names,sorted_items,100)
#
# # now print the results
# # print("\nAbstract:")
# # print(doc)
# print("\nKeywords:")
# for k in keywords:
#     print(k,keywords[k])
