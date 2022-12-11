import dill
import os
import pandas as pd
from rectools.dataset import Dataset

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.tools.utils import prepare_kion_dataset


class PopularModel(BaseRecoModel):

    def __init__(self):
        super().__init__()
        assert os.path.exists("./reco_models/models_raw/pop.dill"), "No model"
        self.pop = dill.load(open("./reco_models/models_raw/pop.dill", 'rb'))
        interactions, users, items = prepare_kion_dataset()
        self.dataset = self.__prepare_dataset(interactions, items)

        self.most_popular_items = list(self.pop.recommend(
            [0],
            dataset=self.dataset,
            k=300,
            filter_viewed=False)["item_id"])

    def recommend(self, user_id, filter_viewed=True, k=10):
        reco_list = list(self.pop.recommend(
            [user_id],
            dataset=self.dataset,
            k=k,
            filter_viewed=filter_viewed)["item_id"])

        return reco_list

    def get_most_popular_items(self):
        return self.most_popular_items

    @staticmethod
    def __prepare_dataset(interactions, items):
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
