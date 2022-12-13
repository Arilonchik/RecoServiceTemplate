from reco_models.models_recomendations.PlugModel import PlugModel
from reco_models.models_recomendations.PopularModel import PopularModel
from reco_models.models_recomendations.UserKnnModel import UserKnnModel
from reco_models.models_recomendations.BlendedUserKnnPopModel import BlendedUserKnnPopular


class ModelValidator:
    def __init__(self):
        self.plug_model = PlugModel()
        self.popular_model = PopularModel(
            model_path="./reco_models/models_raw/pop.dill", pop_type="zip")
        self.user_model = UserKnnModel(
            model_path="./reco_models/models_raw/userknn_50.dill")
        self.blended_user_pop = BlendedUserKnnPopular(self.user_model,
                                               self.popular_model)
        self.model_names = {
            "plug": self.plug_model,
            "popular": self.popular_model,
            "userknn_base": self.user_model,
            "blended_user_pop": self.blended_user_pop
        }

    def get_reco_model(self, model_name: str):
        return self.model_names.get(model_name)
