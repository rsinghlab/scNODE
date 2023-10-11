'''
Description:
    The dummy model: replicates the previous time point as the estimation.
'''

class DummyModel:
    def __init__(self):
        pass

    def predict(self, all_traj_data, train_tps, know_all=False):
        '''
        Predict expression at given timepoints.
        :param all_traj_data: Expression matrices at measured timepoints.
        :param train_tps: A list of measured timepoints
        :param know_all (bool): Whether the model can see all timepoints. If know_all=True, the dummy model will use
                                the eaxct previous timepoint for prediction. Otherwise, it will use the closest available
                                timepoint. For example, if we want to predict t=4 from t=0, 1, 2. If known_all=True,
                                dummy model will replicate expressions at t=3; otherwise, it will replicate t=2.
        :return: Predicted expression at all timepoints.
        '''
        n_tps = len(all_traj_data)
        if not know_all:
            # assume only training data are available
            pred_data = [None]
            last_true = all_traj_data[0]
            for t in range(1, n_tps):
                if t in train_tps:
                    pred_data.append(last_true)
                    last_true = all_traj_data[t]
                else:
                    pred_data.append(last_true)
        else:
            # assume we know the true data of all time points
            pred_data = [None] + [all_traj_data[t] for t in range(n_tps-1)]
        return pred_data
