from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
#visualization
def visualize_data_set(data):
    sns.distplot(data)
    plt.show()
def letters_only(astr): 
    return astr.isalpha()
def count_words(data):
     cv = CountVectorizer(stop_words="english", max_features=500)
     transformed = cv.fit_transform(groups.data) 
     print(cv.get_feature_names())
     sns.distplot(np.log(transformed.toarray().sum(axis=0))) 
     plt.xlabel('Log Count')
     plt.ylabel('Frequency')
     plt.title('Distribution Plot of 500 Word Counts')
     plt.show()
     return transformed

def lets_kmeans(data,target):
    model=KMeans(n_clusters=20,max_iter=1000)
    model.fit(data)
    labels=target
    plt.scatter(labels, model.labels_)
    plt.xlabel('Newsgroup') 
    plt.ylabel('Cluster') 
    plt.show() 

if __name__ == "__main__":
    
    groups = fetch_20newsgroups()
    # visualize_data_set(groups.target)

    # count_words(groups)
    input("press enter to continue")

    print("stemming and lemming it")
    cleaned=[]
    all_n=set(names.words())
    lemmatizer = WordNetLemmatizer()
    for post in groups.data: 
         cleaned.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_n]))
    
    lets_kmeans(count_words(cleaned),groups.target)
    input("press enter to continue")





    
