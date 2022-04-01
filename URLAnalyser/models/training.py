from sklearn.model_selection import RandomizedSearchCV


def tune_hyperparameters(model, hyperparameters, features, labels):
    tuned_model = RandomizedSearchCV(
        estimator=model,
        scoring=["f1", "recall"],
        refit="recall",
        param_distributions=hyperparameters,
        cv=2, n_jobs=-1, verbose=1
    )
    tuned_model.fit(features, labels)
    return model.set_params(**tuned_model.best_params_)
