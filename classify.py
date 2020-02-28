import pandas as pd
import nltk
w = nltk.word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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
