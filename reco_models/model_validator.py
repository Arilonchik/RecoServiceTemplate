from reco_models.models_recomendations.PlugModel import PlugModel
from reco_models.models_recomendations.LightFMModel import LightFMModel


class ModelValidator:
    def __init__(self):
        self.plug_model = PlugModel()
        self.lightfm = LightFMModel()

        self.model_names = {
            "plug": self.plug_model,
            "lightfm": self.lightfm
        }

    def get_reco_model(self, model_name: str):
        return self.model_names.get(model_name)
