import os

import requests
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
from subprocess import Popen, PIPE
from typing import Tuple


from rectools import Columns


class NotEnoughRecoError(Exception):
    def __init__(self, reco_len, message="Not enough in reco list, "):
        self.reco_len = reco_len
        self.message = message + str(self.reco_len)
        super().__init__(self.message)


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


def read_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv('kion_train/interactions.csv')
    users = pd.read_csv('kion_train/users.csv')
    items = pd.read_csv('kion_train/items.csv')

    # rename columns, convert timestamp
    interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                 'total_dur': Columns.Weight},
                        inplace=True)

    interactions['datetime'] = pd.to_datetime(interactions['datetime'])

    return interactions, users, items


def prepare_kion_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists("kion_train.zip"):
        download_dataset()
    if not os.path.exists("kion_train/interactions.csv") or\
            not os.path.exists("kion_train/users.csv") or\
            not os.path.exists("kion_train/items.csv"):
        unzip_dataset()

    interactions, users, items = read_dataset()

    return interactions, users, items


if __name__ == '__main__':
    prepare_kion_dataset()
