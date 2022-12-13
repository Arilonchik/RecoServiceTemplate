import pickle

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from typing import Dict, List


class LightFMModel(BaseRecoModel):
    def __init__(self):
        super().__init__()
        self.offline_model: Dict[int, List] = self.load_pickle("./reco_models/models_raw/offline_lightfm.pkl")
        self.popular: List[int] = self.load_pickle("./reco_models/models_raw/popular.pkl")
    
    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def recommend(self, user_id: int):
        recs = self.offline_model.get(user_id)
        if recs:
            return recs
        return self.popular
        
