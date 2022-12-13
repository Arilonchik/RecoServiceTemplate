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
                watched = self.user_knn_model.get_watched_dict()[user_id]
                additional_reco = self.popular_model.get_most_popular_items()

                for rec in additional_reco:
                    if len(new_reco) == 10:
                        break
                    if rec not in new_reco and rec not in watched:
                        new_reco.append(rec)
                if len(new_reco) < 10:
                    for rec in additional_reco:
                        if len(new_reco) == 10:
                            break
                        if rec not in new_reco:
                            new_reco.append(rec)

                reco_list = new_reco
            assert len(reco_list) == 10, f"Not enough in reco list," \
                                         f" {len(reco_list)}"
            return reco_list

        except KeyError:
            reco_list = self.popular_model.get_most_popular_items()[:10]
            return reco_list
