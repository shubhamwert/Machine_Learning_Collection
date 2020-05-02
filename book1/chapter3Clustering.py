import scipy as sc
from sklearn.feature_extraction.text import CountVectorizer
import os
vectorizer=CountVectorizer(min_df=1)

content = [" I am Gonna do it", " I am Gonna work hard for it"]

X=vectorizer.fit_transform(content)
print("Feature names",vectorizer.get_feature_names())
print("learned vocabulary",X.toarray().transpose())
#from folder chapter3post
clear


vectorizer = CountVectorizer(min_df=1)
DIR="D:\mywork\ML\practiceML\chapter3posts"
posts=[open(os.path.join(DIR,f)).read() for  f in os.listdir(DIR)]
X=vectorizer.fit_transform(posts)
print(vectorizer.get_feature_names())
num_samples, num_features = X.shape

new_post = "imaging databases"
new_post_vector=vectorizer.fit_transform([new_post])
print(new_post_vector)
print(new_post_vector.toarray()) 

