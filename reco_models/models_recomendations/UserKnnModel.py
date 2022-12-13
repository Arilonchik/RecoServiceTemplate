import dill
import os
import pandas as pd
import numpy as np

from collections import Counter
from typing import List

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.tools.utils import prepare_kion_dataset


class UserKnnModel(BaseRecoModel):
    def __init__(self, model_path: str):
        super().__init__()
        assert os.path.exists(model_path), "No model"
        self.model = self.load_dill(model_path)

        self.dataset, users, items = prepare_kion_dataset()

        self.idf, self.idf_dict = self.__prepare_idf()

        train_ids = np.load("./reco_models/models_raw/train_ids.npy",
                            allow_pickle=True)
        train = self.dataset.loc[train_ids]
        self.dataset = train

        self.watched, self.watched_dict = self.prepare_watched()

        self.users_inv_mapping = dict(enumerate(self.dataset['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}
        self.items_inv_mapping = dict(enumerate(self.dataset['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

        self.mapper = self.generate_implicit_recs_mapper(N=50)

    def load_dill(self, path: str):
        with open(path, 'rb') as f:
            model = dill.load(f)
        return model

    def recommend(self, user_id) -> List[int]:
        rco = pd.DataFrame({
            'user_id': [user_id]
        })
        similar_users, similarity = zip(
            *rco['user_id'].map(self.mapper))

        rco_fast = []
        ban_items = {}
        for similar_user, similar in zip(similar_users[0], similarity[0]):
            if similar >= 1:
                continue
            for item_id in self.watched_dict[similar_user]:
                if item_id in ban_items.keys():
                    continue
                ban_items[item_id] = None
                rank_idf = similar * self.idf_dict[item_id]

                new = [user_id, similar_user, similar, item_id, rank_idf]
                rco_fast.append(new)

        rco_fast.sort(key=lambda x: x[-1], reverse=True)

        reco_list = [x[-2] for x in rco_fast][0:10]

        return reco_list

    def generate_implicit_recs_mapper(self, N=50):
        def _recs_mapper(user):
            user_id = self.users_mapping[user]
            recs = self.model.similar_items(user_id, N=N)
            recs.pop(0)
            return [self.users_inv_mapping[user] for user, _ in recs], [sim for
                                                                   _, sim in
                                                                   recs]

        return _recs_mapper

    def prepare_watched(self):
        watched = self.dataset.groupby('user_id').agg({'item_id': list})

        watched_dict = {user_id: items["item_id"] for user_id, items
                        in watched.iterrows()}
        return watched, watched_dict

    def __prepare_idf(self):
        cnt = Counter(self.dataset['item_id'].values)
        idf = pd.DataFrame.from_dict(cnt, orient='index',
                                     columns=['doc_freq']).reset_index()
        # num of documents = num of recommendation list = dataframe shape
        n = self.dataset.shape[0]
        idf['idf'] = idf['doc_freq'].apply(lambda x:
                                           np.log((1 + n) / (1 + x) + 1))

        idf_dict = {items["index"]: items["idf"] for ind, items in
                    idf.iterrows()}
        return idf, idf_dict

    def get_watched_dict(self):
        return self.watched_dict
