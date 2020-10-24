import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def lightgbm_KFold(train, target, params, train_params, categorical_feature, metric, n_splits=5, shuffle=True,
                   random_state=42, use_KFold=True):
    """LightGBM with KFold
    The function trains a LightGBM model using KFold CV

    Arguments:
        train:  pandas DataFrame
            Train dataset. Categorical features must be encoded first.
        target: pandas DataFrame
            Target (label) dataset
        params: dict
            Parameters of LightGBM model. See: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        train_params: dict
            Training parameters of LightGBM model. See: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
        categorical_feature: list of str
            Categorical feature list. Should be a list of str.
        metric: sklearn.metric object
            A sklearn.metric object
        n_splits: int, default = 5
            Number of folds. Must be at least 2.
        shuffle: boolean, default = True
            Whether to shuffle the dataset when using KFold
        random_state: int, default = 42
            Random state of KFold
        use_KFold: boolean, default = True
            Use KFold. Set to True if using Stratified KFold
    Returns:
        models: list of LightGBM model object
            List of LightGBM models, each is trained separately on every fold.
    """
    models = []
    if use_KFold:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        kf_split = kf.split(train)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        kf_split = kf.split(train, target)

    oof_train = np.zeros((len(train)))

    for fold_id, (train_idx, valid_idx) in enumerate(kf_split):
        X_tr = train.iloc[train_idx]
        X_val = train.iloc[valid_idx]
        y_tr = target.iloc[train_idx]
        y_val = target.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_feature)
        lgb_valid = lgb.Dataset(X_val, y_val, categorical_feature=categorical_feature, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            **train_params
        )
        oof_train[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        models.append(model)

    print("Metric: {}".format(metric(target, oof_train)))
    return models

