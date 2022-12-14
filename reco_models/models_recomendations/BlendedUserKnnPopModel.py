from typing import List

from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.models_recomendations.PopularModel import PopularModel
from reco_models.models_recomendations.UserKnnModel import UserKnnModel
from reco_models.tools.utils import NotEnoughRecoError


class BlendedUserKnnPopular(BaseRecoModel):
    def __init__(self, user_knn_model: UserKnnModel,
                 pop_model: PopularModel, k: int = 10) -> None:
        super().__init__()
        self.user_knn_model = user_knn_model
        self.popular_model = pop_model
        self.k = k

    def recommend(self, user_id: int) -> List[int]:
        try:
            reco_list = self.user_knn_model.recommend(user_id)
        except KeyError:
            reco_list = self.popular_model.get_most_popular_items()[:self.k]
            return reco_list

        if len(reco_list) < self.k:
            new_reco = [*reco_list]
            watched = self.user_knn_model.get_watched_dict()[user_id]
            additional_reco = self.popular_model.get_most_popular_items()

            for rec in additional_reco:
                if len(new_reco) == self.k:
                    break
                if rec not in new_reco and rec not in watched:
                    new_reco.append(rec)
            if len(new_reco) < self.k:
                for rec in additional_reco:
                    if len(new_reco) == self.k:
                        break
                    if rec not in new_reco:
                        new_reco.append(rec)

            reco_list = new_reco
        if len(reco_list) != self.k:
            raise NotEnoughRecoError(len(reco_list))

        return reco_list
