import os
import json
import gzip
import pandas as pd
from deep_seo.utils import cleandata, clean_tags

def get_computer_data():
    current_dir =os.path.dirname(__file__)
    raw_data_path = os.path.join(current_dir,'..', 'raw_data')
    data = []
    with gzip.open(os.path.join(raw_data_path,'meta_Computers.json.gz')) as f:
        for l in f:
            data.append(json.loads(l.strip()))
    # convert list into pandas dataframe
    df_comp = pd.DataFrame.from_dict(data)

    def list_to_pd_dataframe(df):

        df3 = df.fillna('')
        df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows
        return df5

    return cleandata(list_to_pd_dataframe(df_comp))


def get_telephone_data():
    """
    This function opens the meta cellphone 100k dataset from the raw_data directory on the project
    also clean the description from the html present
    """
    current_dir =os.path.dirname(__file__)
    raw_data_path = os.path.join(current_dir,'..', 'raw_data')
    df = pd.read_csv(os.path.join(raw_data_path,'meta_cellphone_100k.csv'))
    df_new = df[['category',
                'description',
                'title',
                'brand',
                'feature',
                'imageURLHighRes',
                'main_ranking_3',
                'numb_rankings']]
    df_new['clean_description'] = df_new['description'].map(clean_tags)
    return df_new
    
