from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
import pickle


class LightFMModel(BaseRecoModel):
    def __init__(self):
        super().__init__()
        self.offline_model = pickle.load(open("./reco_models/models_raw/offline_lightfm.pkl", 'rb'))
        self.popular = pickle.load(open("./reco_models/models_raw/popular.pkl", 'rb'))
        
    def recommend(self, user_id):
        try:
            return self.offline_model[user_id]
        except:
            return self.popular