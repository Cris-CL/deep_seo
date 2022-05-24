import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def punc(x):
    for i in string.punctuation:
        x=x.replace(i,"")
    return x

def lower(x):
    x=x.lower()
    return x

def nonumbers(x):
    x="".join(i for i in x if not i.isdigit())
    return x

def stopword(x):
    stop_words=stopwords.words("english") # a list of words that are meaningless to the computer eg: just,will,me.and
    word_token= word_tokenize(x) # split the sentence to each word
    x= [w for w in word_token if not w in stop_words]
    return x

def lemmatize(x):
    lemmatizer=WordNetLemmatizer()
    lemmatized=[lemmatizer.lemmatize(word) for word in x]
    x=" ".join(lemmatized)
    return x
