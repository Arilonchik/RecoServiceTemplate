import pprint

import dill
import os
import pandas as pd
from collections import Counter
import numpy as np
import pprint


from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.tools.utils import prepare_kion_dataset


class UserKnnModel(BaseRecoModel):
    def __init__(self):
        super().__init__()
        assert os.path.exists("./reco_models/models_raw/userknn_50.dill"), "No model"
        self.model = dill.load(open("./reco_models/models_raw/userknn_50.dill", 'rb'))

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

    def recommend(self, user_id):
        rco = pd.DataFrame({
            'user_id': [user_id]
        })
        similar_users, similarity = zip(
            *rco['user_id'].map(self.mapper))

        rco_fast = []
        ban_items = {}
        for similar_user, similar in zip(similar_users[0], similarity[0]):
            for item_id in self.watched_dict[similar_user]:
                if item_id in ban_items.keys():
                    continue
                ban_items[item_id] = None
                rank_idf = similar * self.idf_dict[item_id]

                new = [user_id, similar_user, similar, item_id, rank_idf]
                rco_fast.append(new)

        rco_fast.sort(key=lambda x: x[-1], reverse=True)

        reco_list = [x[-2] for x in rco_fast][0:10]
        # rco['similar_user_id'], rco['similarity'] = zip(
        #     *rco['user_id'].map(self.mapper))
        #
        # rco = rco.set_index('user_id').apply(pd.Series.explode).reset_index()
        # # rco = rco[~(rco['similarity'] >= 1)]
        # rco = rco.merge(self.watched, left_on=['similar_user_id'],
        #                 right_on=['user_id'], how='left')
        #
        # rco = rco.explode('item_id')

        # # rco = rco.sort_values(['user_id', 'similarity'], ascending=False)
        #
        # rco = rco.drop_duplicates(['user_id', 'item_id'], keep='first')
        #
        # rco = rco \
        #     .merge(
        #     self.idf[['index', 'idf']],
        #     left_on='item_id',
        #     right_on='index',
        #     how='left') \
        #     .drop(['index'], axis=1)

        # rco['rank_idf'] = rco['similarity'] * rco['idf']
        # rco = rco.sort_values(['user_id', 'rank_idf'], ascending=False)
        #
        # reco_list = list(rco["item_id"])[0:10]
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
        idf['idf'] = idf['doc_freq'].apply(lambda x: np.log((1 + n) / (1 + x) + 1))

        idf_dict = {items["index"]: items["idf"] for ind, items in
                    idf.iterrows()}
        return idf, idf_dict

