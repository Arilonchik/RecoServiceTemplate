from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel


class PlugModel(BaseRecoModel):

    def __init__(self):
        super().__init__()

    def recommend(self, user_id):
        reco_list = list(range(10))
        return reco_list
