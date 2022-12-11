import dill
import os

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.tools.utils import prepare_kion_dataset


class PopularModel(BaseRecoModel):

    def __init__(self):
        super().__init__()
        assert os.path.exists("./reco_models/models_raw/pop.dill"), "No model"
        self.pop = dill.load(open("./reco_models/models_raw/pop.dill", 'rb'))
        self.dataset = prepare_kion_dataset()

    def recommend(self, user_id):
        reco_list = list(self.pop.recommend(
            [user_id],
            dataset=self.dataset,
            k=10,
            filter_viewed=True)["item_id"])

        return reco_list
