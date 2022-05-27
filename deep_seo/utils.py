import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

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


def clean_tags(txt):
    return txt.replace('<br>','').replace('<b>','').replace('</br>','').replace('</b>','').replace('"','').replace("'",'').replace(',','').replace('[','').replace(']','').replace('&amp','')

def featureclean(cols):
    return cols.replace("[]","na").strip("[]'").lower()


def super_popular_brand(df):
    df_brand=df[["brand","main_ranking_3"]].copy()
    count_df=pd.DataFrame(df_brand[["brand"]].value_counts())
    min_df=pd.DataFrame(df.groupby("brand").min()["main_ranking_3"])
    df3=count_df.merge(min_df,on="brand").sort_values("main_ranking_3").reset_index()
    df3=df3.rename(columns={0:"value_counts"})

    a=[]

    df4=df3[df3["value_counts"]>1]
    df_tmp = df4.query("0 < main_ranking_3 <= 100")     #[df4["main_ranking_3"]<= 100]
    for brandpop in df_tmp["brand"]:
        a.append(brandpop)
    return a

def popular_brand(df):
    df_brand=df[["brand","main_ranking_3"]].copy()
    count_df=pd.DataFrame(df_brand[["brand"]].value_counts())
    min_df=pd.DataFrame(df.groupby("brand").min()["main_ranking_3"])
    df3=count_df.merge(min_df,on="brand").sort_values("main_ranking_3").reset_index()
    df3=df3.rename(columns={0:"value_counts"})

    a=[]

    df4=df3[df3["value_counts"]>1]
    df_tmp = df4.query("100 < main_ranking_3 <= 500")
    for brandpop in df_tmp["brand"]:
        a.append(brandpop)
    return a

def good_brand(df):
    df_brand=df[["brand","main_ranking_3"]].copy()
    count_df=pd.DataFrame(df_brand[["brand"]].value_counts())
    min_df=pd.DataFrame(df.groupby("brand").min()["main_ranking_3"])
    df3=count_df.merge(min_df,on="brand").sort_values("main_ranking_3").reset_index()
    df3=df3.rename(columns={0:"value_counts"})

    a=[]

    df4=df3[df3["value_counts"]>1]
    df_tmp = df4.query("500 < main_ranking_3 <= 1000")
    for brandpop in df_tmp["brand"]:
        a.append(brandpop)
    return a

def normal_brand(df):
    df_brand=df[["brand","main_ranking_3"]].copy()
    count_df=pd.DataFrame(df_brand[["brand"]].value_counts())
    min_df=pd.DataFrame(df.groupby("brand").min()["main_ranking_3"])
    df3=count_df.merge(min_df,on="brand").sort_values("main_ranking_3").reset_index()
    df3=df3.rename(columns={0:"value_counts"})

    a=[]

    df4=df3[df3["value_counts"]>1]
    df_tmp = df4.query("1000 < main_ranking_3 <= 1500")
    for brandpop in df_tmp["brand"]:
        a.append(brandpop)
    return a

def bad_brand(df):
    df_brand=df[["brand","main_ranking_3"]].copy()
    count_df=pd.DataFrame(df_brand[["brand"]].value_counts())
    min_df=pd.DataFrame(df.groupby("brand").min()["main_ranking_3"])
    df3=count_df.merge(min_df,on="brand").sort_values("main_ranking_3").reset_index()
    df3=df3.rename(columns={0:"value_counts"})

    a=[]

    df4=df3[df3["value_counts"]>1]
    df_tmp = df4.query("main_ranking_3 > 1500")
    for brandpop in df_tmp["brand"]:
        a.append(brandpop)
    return a

def brand_categorical(df):
    super_popular = super_popular_brand(df)
    popular_brands = popular_brand(df)
    good = good_brand(df)
    normal = normal_brand(df)
    bad_brands = bad_brand(df)

    def which_brand(brand_name):
        if brand_name in super_popular:
            return 1
        elif brand_name in popular_brands:
            return 2
        elif brand_name in good:
            return 3
        elif brand_name in normal:
            return 4
        elif brand_name in bad_brands:
            return 5
        ## if is not in any of the lists by default puts them in the normal category
        return 4
    df['brand_cat'] = df['brand'].map(which_brand)
    return df
