from reco_models.models_recomendations.PlugModel import PlugModel


class ModelValidator:
    def __init__(self):
        self.plug_model = PlugModel()
        self.model_names = {
            "plug": self.plug_model
        }

    def get_reco_model(self, model_name):
        if model_name not in self.model_names.keys():
            return None
        else:
            return self.model_names[model_name]
