from reco_models.models_recomendations.BaseRecoModel import BaseRecoModel
from reco_models.models_recomendations.PopularModel import PopularModel
from reco_models.models_recomendations.UserKnnModel import UserKnnModel


class BlendedUserKnnPopular(BaseRecoModel):
    def __init__(self, user_knn_model: UserKnnModel, pop_model: PopularModel):
        super().__init__()
        self.user_knn_model = user_knn_model
        self.popular_model = pop_model

    def recommend(self, user_id):
        try:
            reco_list = self.user_knn_model.recommend(user_id)
            if len(reco_list) < 10:
                new_reco = [*reco_list]
                additional_reco = self.popular_model.recommend(user_id, k=20)

                for rec in additional_reco:
                    if len(new_reco) == 10:
                        break
                    if rec not in new_reco:
                        new_reco.append(rec)

                reco_list = new_reco
            return reco_list

        except KeyError as e:
            print("Cold user detected")
            reco_list = self.popular_model.recommend(0, filter_viewed=False)
            return reco_list
