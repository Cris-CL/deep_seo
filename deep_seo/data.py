import os
import json
import gzip
import pandas as pd
from deep_seo.utils import cleandata

def get_computer_data():
    data = []
    with gzip.open('../raw_data/meta_Computers.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    # convert list into pandas dataframe
    df_comp = pd.DataFrame.from_dict(data)

    def list_to_pd_dataframe(df):

        df3 = df.fillna('')
        df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows
        return df5
    return cleandata(list_to_pd_dataframe(df_comp))
