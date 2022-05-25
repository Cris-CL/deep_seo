import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

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

def full_text_processing(columns):
    a=""
    for i in columns:
        a=a+i
    a=punc(a)
    a=lower(a)
    a=nonumbers(a)
    a=stopword(a)
    a=lemmatize(a)
    b=[a]
    vectorizer = TfidfVectorizer(max_features = 100)
    X = vectorizer.fit_transform(b) #brand
    return X

def cleandata(df5):
    df5.description=df5.description.map(lambda x: str(x))
    df5.description=df5.description.str.strip("[]'")
    df5.drop(columns=['feature', 'tech1', 'also_buy', 'price', 'also_view', 'tech2','details', 'similar_item',"main_cat"],inplace=True)
    df5['brand_cat']=df5.brand.map(lambda x : 0 if len(x) <1 else 1)
    df5['rank1']=df5['rank'].map(lambda x: str(x).strip("[]'"))
    df5["rank1"]=df5["rank1"].map(lambda x: str(x))
    df5["rank1"]=df5["rank1"].str.replace(pat='>#',repl='', regex=False)
    df5["rank1"]=df5["rank1"].map(lambda x: x.replace(",",""))
    df5["rank1"]=df5["rank1"].map(lambda x: x.split("in"))
    df5["rank1"] = df5["rank1"].map(lambda x: x[0])
    df5["title_count"] = df5["title"].map(lambda x: len(x))
    df5["desc_count"] = df5["description"].map(lambda x: len(x))
    df5["img_count"] = df5["image"].map(lambda x: len(x))
    df6=df5[df5["rank1"]!='']
    df6["rank1"]=df6["rank1"].astype("int64")
    df6["rank_cat"]=df6["rank1"].map(lambda x: 1 if x<473359 else 0)
    return df6
