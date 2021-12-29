import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor

import pickle

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

class ModelClimbers():
    """
    Takes climbing data
    """
    def __init__(self,path: str = os.path.join("data", "processed", "imputed_data.csv"), df: pd.DataFrame = False) -> None:
        """
        Uploads the csv of imputed climbing data. By default uses a path to the csv, otherwise, a dataframe can be used.
        """

        if df:
            print("Using dataframe gather data.")
            self.data = df

        elif path:
            print("Using path to csv to upload data.")
            if not os.path.exists(path):
                print(f"The path {path} does not exist!")
                return
            else: 
                self.data = pd.read_csv(path).iloc[:,1:]
        else:
            print("Must provide a path or dataframe!")
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_std = None
        self.X_test_std = None
        self.models = {}
        self.cv_scores = None

    def make_train_test(self):
        """
        Shuffles the data and performs a train/test split.
        """

        X = self.data.drop(columns=['max_grade'])
        y = self.data['max_grade']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def make_standardized(self, save_scaler: bool = False, path: str = os.path.join("models", "std_scaler.pkl")):
        """
        Scales the test data, can also optionally save the standard scaler. 
        """

        numeric_feats = ['height', 'weight', 'years_climbing', 'age']

        col_trans = ColumnTransformer([('std_scale_num', StandardScaler(), numeric_feats)], remainder='passthrough')

        std_scaler = col_trans.fit(self.X_train)

        X_train_std = std_scaler.transform(self.X_train)
        X_test_std = std_scaler.transform(self.X_test)

        self.X_train_std = X_train_std
        self.X_test_std = X_test_std

        if save_scaler:
            if os.path.exists(path):
                print(f"Standard scaler already exists at {path}")
            else:
                pickle.dump(std_scaler, open(path,'wb'))
                print(f"Saved standard scaler to {path}")
    
    def linear_regression(self):
        """
        Trains a linear regression model.
        """

        model = linear_model.LinearRegression()
        model_fit = model.fit(self.X_train_std, self.y_train)

        cv = cross_validate(model, 
            self.X_train_std, 
            self.y_train,
            cv=10,
            scoring=("neg_root_mean_squared_error", "r2"))

        
        lin_reg = {
            "model": model_fit,
            "rmse_cv": cv["test_neg_root_mean_squared_error"],
            "r2_cv": cv["test_r2"]
        }

        self.models["linear_regression"] = lin_reg
    
    @ignore_warnings(category=ConvergenceWarning)
    def elastic_net(self):
        """
        Trains an elastic net model with hyperparameter tuning.
        """

        model = linear_model.ElasticNet()

        param_grid = {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": np.arange(0.0, 1.0, 0.1)
        }

        print("Hyper parameter tuning elastic net model...")
        grid_cv = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", cv=10)

        model_fit = grid_cv.fit(self.X_train_std, self.y_train)
        best_params = model_fit.best_estimator_.get_params()

        print("Cross validating elastic net model...")
        cv = cross_validate(linear_model.ElasticNet(**best_params), 
            self.X_train_std, 
            self.y_train,
            cv=10,
            scoring=("neg_root_mean_squared_error", "r2"))

        
        elas_net = {
            "model": model_fit,
            "rmse_cv": cv["test_neg_root_mean_squared_error"],
            "r2_cv": cv["test_r2"]
        }

        self.models["elastic_net"] = elas_net

    def xg_boost(self):
        """
        Trains an xgboost model and tunes the hyperparameters.
        """
        
        model = XGBRegressor()
        #model_fit = XGBRegressor().fit(data["X_train_std"], data["y_train"])

        param_grid = {
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.4, 0.7, 1],
            'colsample_bytree': [0.2, 0.6, 1],
            'n_estimators' : [100, 200, 500]
        }

        print("Hyper parameter tuning xgboost model...")
        grid_cv = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", cv=10)

        model_fit = grid_cv.fit(self.X_train_std, self.y_train)
        best_params = model_fit.best_estimator_.get_params()

        print("Cross validating xgboost model...")
        cv = cross_validate(XGBRegressor(**best_params), 
            self.X_train_std, 
            self.y_train,
            cv=10,
            scoring=("neg_root_mean_squared_error", "r2"))

        
        xgb = {
            "model": model_fit,
            "rmse_cv": cv["test_neg_root_mean_squared_error"],
            "r2_cv": cv["test_r2"]
        }

        self.models["xgboost"] = xgb


    def save_plot_cv_scores(self, path: str = os.path.join("figures", "cross_validation_scores.png")):
        """
        Saves a dataframe of cv scores for each model and figure plotting the scores.
        """

        df = pd.DataFrame()
        for mod in self.models.keys():
            cur_df = pd.DataFrame(
                {
                    "rmse": np.abs(self.models[mod]["rmse_cv"]),
                    "r2": self.models[mod]["r2_cv"],
                    "model": mod
                }
            )

            df = pd.concat([df, cur_df])
        
        self.cv_scores = df

        scores = ["rmse", "r2"]
        
        sns.set_style("whitegrid")
        plt.subplots(figsize=(16, 14), dpi=200)
        plt.margins(x=0.5, y=0.5)
        for i, col in enumerate(scores):
            plt.subplot(1, len(scores), i+1)
            sns.boxplot(y= col, x="model", data = df)
            plt.ylabel(col, fontsize=16,rotation='horizontal', labelpad=30)
            plt.xlabel("", labelpad=30)
            plt.xticks(fontsize=16)

        plt.suptitle('10-Fold Cross Validation Scores', fontsize=20)
        plt.savefig(path)
        
    def pickle_models(self, path = "models"):
        """
        Saves all models as pickle file. 
        """

        for model in self.models.keys():
            filename = f"{model}.pkl"
            filepath = os.path.join(path, filename)
            pickle.dump(model, open(filepath, 'wb'))
            print(f"Saved {model} to {filepath}.")


def main():
    modelobject = ModelClimbers()
    modelobject.make_train_test()
    modelobject.make_standardized(save_scaler= True)
    modelobject.linear_regression()
    modelobject.elastic_net()
    modelobject.xg_boost()
    modelobject.save_plot_cv_scores()
    modelobject.pickle_models()

if __name__ == '__main__':
    main()