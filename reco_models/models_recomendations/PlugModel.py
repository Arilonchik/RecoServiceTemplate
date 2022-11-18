from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel


class PlugModel(BaseRecoModel):

    def __init__(self):
        super().__init__()

    def recommend(self, user_id):
        reco_list = [i for i in range(10)]
        return reco_list
