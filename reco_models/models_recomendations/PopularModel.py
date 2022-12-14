import dill
import os
import pandas as pd
import numpy as np
from scipy.stats import mode
from rectools.dataset import Dataset
from typing import List, Tuple
from enum import Enum

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.tools.utils import prepare_kion_dataset


class PopularType(Enum):
    SIMPLE = "simple"
    ZIP = "zip"


class PopularModel(BaseRecoModel):

    def __init__(self, model_path: str,
                 pop_type: PopularType = PopularType.SIMPLE):
        super().__init__()
        if pop_type == "simple":
            assert os.path.exists(model_path), "No model"
            self.pop = dill.load(open(model_path, 'rb'))
            interactions, users, items = prepare_kion_dataset()
            self.dataset = self.__prepare_dataset(interactions, items)

            self.most_popular_items = list(self.pop.recommend(
                [0],
                dataset=self.dataset,
                k=1298,
                filter_viewed=False)["item_id"])

        if pop_type == "zip":
            interactions, users, items = prepare_kion_dataset()
            self.dataset = self.__prepare_dataset(interactions, items)
            # get csr matrix from interactions
            matrix = self.dataset.get_user_item_matrix()
            item_set, covered_users = self.__get_top_items_covered_users(
                matrix, n_users=900000)
            self.most_popular_items = list(
                self.dataset.item_id_map.convert_to_external(item_set)
            )

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        reco_list = self.most_popular_items[:k]

        return reco_list

    def get_most_popular_items(self) -> List[int]:
        return self.most_popular_items

    @staticmethod
    def __get_top_items_covered_users(
            matrix, n_users: int = 1000) -> Tuple[list, np.array]:

        assert matrix.format == 'csr'

        item_set = []
        covered_users = np.zeros(matrix.shape[0],
                                 dtype=bool)
        while covered_users.sum() < n_users:
            top_item = mode(matrix[~covered_users].indices)[0][
                0]
            item_set.append(top_item)
            covered_users += np.maximum.reduceat(matrix.indices == top_item,
                                                 matrix.indptr[:-1],
                                                 dtype=bool)
        return item_set, covered_users

    @staticmethod
    def __prepare_dataset(interactions: pd.DataFrame,
                          items: pd.DataFrame) -> Dataset:
        _, bins = pd.qcut(items["release_year"], 10, retbins=True)

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
