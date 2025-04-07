import pandas  as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing)

data = pd.DataFrame(housing.data, columns = housing.feature_names)
data["price"] =  housing.target
data.head(10)

## split the data for the model 

from urllib.parse import urlparse
X = data.drop('price', axis=1)
y = data["price"]
print("created X y ")

#Hyperparameter tuning using GridSearchCV
def HyperparmeterTuning(X_trian,y_train ,param_grid):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf , param_grid = param_grid, cv = 5 , n_jobs = -1, verbose = 2 , scoring = "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search

print("Hyperparameter tuning using GridSearchCV")


#spiting the data into train and test
print("Splitting the data into train and test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#infer the signature of the model
print("Infer the signature of the model")

from mlflow.models import infer_signature
signature = infer_signature(X_train, y_train)

param_grid = {
   "n_estimators": [100, 200],
   "max_depth": [5, 10], 
   "max_features": [0.5, 0.9],
   "min_samples_split": [2, 5],   
   "min_samples_leaf": [1, 2] 
}

# start the mlflow run 
with mlflow.start_run():
    grid_search = HyperparmeterTuning(X_train, y_train, param_grid)



    # get the best model 
    best_model = grid_search.best_estimator_

    # evaluate the best model 
    y_pred =  best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # log the best model and the metrics
    mlflow.log_param("n_estimators", grid_search.best_params_["n_estimators"])
    mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
    mlflow.log_param("min_sample_leaf", grid_search.best_params_["min_samples_leaf"])
    mlflow.log_param("min_samples_split", grid_search.best_params_["min_samples_split"])
    mlflow.log_metric("mse", mse)
  
    # Tracking URI
    mlflow.set_tracking_uri (uri="http://127.0.0.1:5000")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(best_model, "model", registered_model_name = "Best Random Forest Model", signature = signature)
    else:
        mlflow.sklearn.log_model(best_model, "model", signature = signature)

print(f" Best Hyperparameters: {grid_search.best_params_}")
print(f" MSE: {mse}")
