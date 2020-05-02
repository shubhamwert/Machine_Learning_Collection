import glob,os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
def letter_only(sstr):
    return sstr.isalpha()

def clean_text(docs):
    
    cleaned=[]
    for doc in docs:
        cleaned.append(''.join([lemmatizer.lemmatize(word.lower())
            for word in str(doc).split()
                if letter_only(word)
                    and word not in all_names]))
    return cleaned

def get_label_index(labels):
    label_index=defaultdict(list)
    print(label_index)
    for idx,label in enumerate(labels):
        label_index(label).append(idx)
    return label_index


if __name__ == "__main__":
    path=r'''D:\mywork\ML\ML git upload\book2\enron1'''
    email=[]
    labels=[]
    for filename in glob.glob(os.path.join(path,'spam\*.txt')):
            with open(filename,'r', encoding="ISO-8859-1") as infile:
                email.append(infile.read())
                labels.append(1)
    for filename in glob.glob(os.path.join(path,'ham\*.txt')):
            with open(filename,'r', encoding="ISO-8859-1") as infile:
                email.append(infile.read())
                labels.append(0)    

    print(len(email),"\n",len(labels))
    all_names=set(names.words())
    lemmatizer = WordNetLemmatizer()
    cleaned_email=clean_text(email)
    cv = CountVectorizer(stop_words="english", max_features=500)  
    term_docs = cv.fit_transform(cleaned_email)
    # print(term_docs) 
    feature_mapping = cv.vocabulary_ 
    #print(feature_mapping)
    label_index=get_label_index(labels)
    print(label_index)



