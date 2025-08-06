import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.utils.validation import check_array
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def day_month_encoding(id):
    month_int= {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    day = id["day"]
    try:
        return datetime.date(2024, month, day).timetuple().tm_yday
    except ValueError:
        return datetime.date(2024, month, min(day, 29)).timetuple().tm_yday
    
def day_month_encoding_pipeline(df):
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    df = df.copy()
    df["month_int"] = df["month"].map(month_map)
    result = []

    for day, month in zip(df["day"], df["month_int"]):
        try:
            doy = datetime.date(2024, month, day).timetuple().tm_yday
        except ValueError:
            doy = datetime.date(2024, month, min(day, 28)).timetuple().tm_yday
        result.append(doy)

    return np.array(result).reshape(-1, 1)


def month_sin_encoding(id):
    month_int = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    return np.sin(2 * np.pi * month / 12)

def month_cos_encoding(id):
    month_int = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    return np.cos(2 * np.pi * month / 12)


def log_encoding(dataframe):
    df = dataframe.copy()
    return df.apply(lambda x : np.sign(x) * np.log1p(abs(x))).to_numpy()
 
 
class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.input_features_ = X.columns if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return log_encoding(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.input_features_ is not None:
                input_features = self.input_features_
            else:
                input_features = [f"log_{i}" for i in range(self.n_features_in_)]
        return np.array([f"log_{name}" for name in input_features])
        
class DayOfYearTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.input_features_ = X.columns if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return day_month_encoding_pipeline(X)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(["day_of_the_year"])


class KNN_useless_features(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_splits=5, random_state=42):
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        print(f"KNN_useless_features received X with shape: {X.shape}")
        print(f"First rows:\n{X[:5]}")
        
        self.imputer = SimpleImputer(strategy="mean")
        X_imp = self.imputer.fit_transform(X)

        y = np.array(y)

        self.knn_models = []
        self.oof_predictions = np.zeros(X_imp.shape[0])

       
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        
        for train_idx, test_idx in kf.split(X_imp):
            knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")
            knn.fit(X_imp[train_idx], y[train_idx])
            self.oof_predictions[test_idx] = knn.predict(X_imp[test_idx])
            self.knn_models.append(knn)

        return self
        
    def transform(self, X):
        
        X_imp = self.imputer.transform(X)

        
        all_preds = np.vstack([model.predict(X_imp) for model in self.knn_models]).T
        return np.mean(all_preds, axis=1).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return [f"knn_oof_mean"]
    
class KNN_useless_features2(BaseEstimator, TransformerMixin):
    def __init__(self, features, n_neighbors=5, n_splits=5, random_state=42):
        self.features = features 
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X_weak = X[self.features].copy()

        self.knn_models = []
        self.oof_predictions = np.zeros(len(X_weak.index))
        self.imputer = SimpleImputer(strategy="mean")
        X_weak = self.imputer.fit_transform(X_weak)
        
        skf = KFold(n_splits = self.n_splits, random_state=self.random_state, shuffle=True)
        
        for train_index, test_index in skf.split(X_weak, y):
            X_weak_train = X_weak[train_index]
            Y_train = y[train_index]

            X_weak_test = X_weak[test_index]

            k_ngb = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="uniform", algorithm="kd_tree", n_jobs=-1)
            k_ngb = k_ngb.fit(X_weak_train, Y_train)
            self.oof_predictions[test_index] = k_ngb.predict(X_weak_test)
            self.knn_models.append(k_ngb)
            print("knn_fitted")

        return self
    
    def transform(self, X):
        X_wk = X[self.features].copy()
        X_wk = self.imputer.transform(X_wk)

        all_pred = np.zeros(shape=(X_wk.shape[0], self.n_splits))

        for i, model in enumerate(self.knn_models):
            all_pred[:, i] = model.predict(X_wk)

        return np.mean(all_pred, axis=1).reshape(X_wk.shape[0],1)
    
    def get_feature_names_out(self, input_features=None):
        return [f"knn_oof_mean__{'_'.join(self.features)}"]


class RF_useless_features(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, n_estimators=100, max_depth=9, n_splits=25, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.random_state = random_state
        self.features= features

    def fit(self, X, y):

        if y is None:
            raise ValueError("y cannot be None for RF_useless_features")
        if self.features is not None:
            X = X[self.features].copy()

        self.imputer = SimpleImputer(strategy="mean")
        X_imp = self.imputer.fit_transform(X)

        y = np.array(y)
        self.rf_models = []
        self.oof_predictions = np.zeros(X_imp.shape[0])

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X_imp):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_imp[train_idx], y[train_idx])
            self.oof_predictions[test_idx] = rf.predict(X_imp[test_idx])
            self.rf_models.append(rf)

        return self

    def transform(self, X):
        if self.features is not None:
            X = X[self.features].copy()
        
        X_imp = self.imputer.transform(X)
        all_preds = np.vstack([model.predict(X_imp) for model in self.rf_models]).T
        return np.mean(all_preds, axis=1).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return ["rf_oof_mean"]
    


class WrapperWithY(BaseEstimator, TransformerMixin):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def fit(self, X, y=None):
        self.wrapped = self.wrapped.fit(X, y)
        return self

    def transform(self, X):
        return self.wrapped.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            n_features = getattr(self.wrapped, 'n_features_in_', None)
            if n_features is not None:
                input_features = [f'x{i}' for i in range(n_features)]
            else:
                return np.array([])

        if hasattr(self.wrapped, "get_feature_names_out"):
            return self.wrapped.get_feature_names_out(input_features)
        elif hasattr(self.wrapped, "get_support"):
            support = self.wrapped.get_support()
            return np.array(input_features)[support]
        else:

            return input_features
    

#needs to be wrapped with WrapperWithY class when the pipeline doesn't directly pass him labels, but only x 
# i.e when utilizing the make_pipeline() function
class KneighborsRegressorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
    
    def fit(self, X, y=None):
        self.model = KNeighborsRegressor(
        n_neighbors=self.n_neighbors,
        weights=self.weights,
        algorithm=self.algorithm 
            )
        
        self.model.fit(X, y)
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            print("Model not fitted, wrap it with 'WrapperWithY' class")
    
        return self
    
    def transform(self, X):
        predictions = self.model.predict(X)
        if predictions.ndim == 1:
            predictions = predictions(-1,1)  #ensures 2D output
        return predictions
    


    #HOML3 version of a generical regression transformer using MetaEstimatorMixin
    '''
    Rather than restrict ourselves to k-Nearest Neighbors regressors, 
    let's create a transformer that accepts any regressor. 
    For this, we can extend the `MetaEstimatorMixin` 
    and have a required `estimator` argument in the constructor. 
    The `fit()` method must work on a clone of this estimator, 
    and it must also save `feature_names_in_`. 
    The `MetaEstimatorMixin` will ensure that `estimator` is listed as a required parameters, 
    and it will update `get_params()` and `set_params()` to make the estimator's hyperparameters available for tuning. 
    Lastly, we create a `get_feature_names_out()` method: the output column name is the ...
    '''
class FeatureFromRegressor(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        check_array(X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self  # always return self!
    
    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_class_name = self.estimator_.__class__.__name__
        estimator_short_name = estimator_class_name.lower().replace("_", "")
        return [f"{estimator_short_name}_prediction_{i}"
                for i in range(n_outputs)]

class StandardScalerClone(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.features_names_in_ = np.array(X.columns)
        
            self.mean_ = []
            self.std_ = []
            for column in self.columns_:
                if self.with_mean:
                    self.mean_.append(X[column].mean())
                else:
                    self.mean_.append(0)

                if self.with_std:
                    if X[column].std() != 0:
                        self.std_.append(X[column].std())
                    else:
                        self.std_.append(1)
                else:
                    self.std_.append(1)
        elif isinstance(X, np.ndarray):

            X_array = X.copy()
            self.mean_ = []
            self.std_ = []
            self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(X_array.shape[1])
            self.std_ = np.std(X, axis=0) if self.with_std else np.ones(X_array.shape[1])
            self.std_ = np.where(self.std_ == 0, 1, self.std_)
                
        else:
            raise ValueError("Input must be pandas DataFrame or numpy ndarray")
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_trans = X.copy()
            for column, mean, std in zip(self.columns_, self.mean_, self.std_):
                X_trans[column] = (X_trans[column] - mean) / std
        
            return X_trans
        elif isinstance(X, np.ndarray):
            return (X - self.mean_) / self.std_

    
    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for column, std, mean in zip(self.columns_, self.std_, self.mean_):
                X_inverse[column] = X_inverse[column] * std + mean
            return X_inverse
        
        elif isinstance(X, np.ndarray):
            X_inverse= X.copy()
            for i, (std, mean) in enumerate(zip(self.std_, self.mean_)):
                X_inverse[:, i] = X_inverse[:, i] * std + mean
            return X_inverse


