import os

import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import mode
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
from subprocess import Popen, PIPE

from rectools import Columns
from rectools.dataset import Dataset


def download_dataset():
    # download dataset by chunks
    url = "https://storage.yandexcloud.net/itmo-recsys-public-data/kion_train.zip"

    req = requests.get(url, stream=True)

    with open('kion_train.zip', "wb") as fd:
        total_size_in_bytes = int(req.headers.get('Content-Length', 0))
        progress_bar = tqdm(desc='kion dataset download',
                            total=total_size_in_bytes, unit='iB',
                            unit_scale=True)
        for chunk in req.iter_content(chunk_size=2 ** 20):
            progress_bar.update(len(chunk))
            fd.write(chunk)


def unzip_dataset():
    cmd = "unzip kion_train.zip"
    with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1,
               universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end=" ")
        for line in p.stderr:
            print(line, end=" ")


def read_dataset():
    interactions = pd.read_csv('kion_train/interactions.csv')
    users = pd.read_csv('kion_train/users.csv')
    items = pd.read_csv('kion_train/items.csv')

    # rename columns, convert timestamp
    interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                 'total_dur': Columns.Weight},
                        inplace=True)

    interactions['datetime'] = pd.to_datetime(interactions['datetime'])

    _, bins = pd.qcut(items["release_year"], 10, retbins=True)
    labels = bins[:-1]

    year_feature = pd.DataFrame(
        {
            "id": items["item_id"],
            "value": pd.cut(items["release_year"], bins=bins,
                            labels=bins[:-1]),
            "feature": "release_year",
        }
    )

    items["genre"] = items["genres"].str.split(",")

    genre_feature = items[["item_id", "genre"]].explode("genre")
    genre_feature.columns = ["id", "value"]
    genre_feature["feature"] = "genre"

    item_feat = pd.concat([genre_feature, year_feature])
    item_feat = item_feat[item_feat['id'].isin(interactions['item_id'])]

    dataset = Dataset.construct(
        interactions_df=interactions,
        user_features_df=None,
        item_features_df=item_feat,
        cat_item_features=['genre', 'release_year']
    )

    return dataset


def prepare_kion_dataset():
    if not os.path.exists("kion_train.zip"):
        download_dataset()
    if not os.path.exists("kion_train/interactions.csv") or\
            not os.path.exists("kion_train/users.csv") or\
            not os.path.exists("kion_train/items.csv"):
        unzip_dataset()

    print("Construct dataset")
    return read_dataset()


if __name__ == '__main__':
    prepare_kion_dataset()
