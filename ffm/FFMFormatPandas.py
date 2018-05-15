import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_classification

'''
A sci-kit learn inspired script to convert pandas dataframes into libFFM data.

The script is fairly hacky (hey thats Kaggle) and takes a little while to run a huge dataset.
The key to using this class is setting up the features dtypes correctly for output (ammend transform to suit your needs)

Example below

'''


class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            if df[col].dtype.kind == 'O':
                if type(df[col].iloc[0]) is dict:
                    vals = set()
                    df[col].apply(lambda d: [vals.add(k) for k, v in d.items()])
                else:
                    vals = df[col].unique()
                for val in vals:
                    if pd.isnull(val):
                        continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            if col_type.kind ==  'O':
                if type(val) is dict:
                    for k,v in val.items():
                        name = '{}_{}'.format(col, k)
                        ffm.append((self.field_index_[col], self.feature_index_[name], v))
                else:
                    name = '{}_{}'.format(col, val)
                    ffm.append((self.field_index_[col], self.feature_index_[name],1))
            elif col_type.kind == 'i' or col_type.kind == 'f':
                ffm.append((self.field_index_[col], self.feature_index_[col], val))
        return ffm

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})
