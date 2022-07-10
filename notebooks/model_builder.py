from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

def build_pipeline(params) -> Pipeline:
    """
        Builds pipeline.
        :params dict: all hyperparameters of the model
        :return Pipeline: returns an sklearn Pipeline object 
    """

    transformers = []

    bool_pipeline = Pipeline(steps=[
        ("cast_type", FunctionTransformer(to_object)),
        ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers.append(("boolean", bool_pipeline, 
                        ["Dependents", "PaperlessBilling", "Partner", "PhoneService", "SeniorCitizen"]))

    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(to_numeric)),
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    transformers.append(("numerical", numerical_pipeline, 
                        ["AvgPriceIncrease", "Contract", "MonthlyCharges", "NumOptionalServices", "TotalCharges", "tenure"]))

    one_hot_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("onehot", one_hot_pipeline, 
                        ["DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", \
                        "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV", "TechSupport", "gender"]))

    model = _build_model(params)

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
        ("standardizer", StandardScaler()),
        ("XGBClassifier", model)
    ])


def _build_model(params):

    if 'max_depth' in params: 
        # hyperopt supplies values as float but must be int
        params['max_depth']=int(params['max_depth'])   
    if 'min_child_weight' in params: 
        # hyperopt supplies values as float but must be int
        params['min_child_weight']=int(params['min_child_weight']) 
    if 'max_delta_step' in params: 
        # hyperopt supplies values as float but must be int
        params['max_delta_step']=int(params['max_delta_step']) 
        
    # all other hyperparameters are taken as given by hyperopt
    xgb_classifier = XGBClassifier(**params)

    return xgb_classifier


def train_model(params, X_train, y_train):
    """ 
    Function that calls the pipeline to train a model and Tune it with HyperOpt 

    :params dict: all hyperparameters of the model
    :return dict: return a dictionary that contains a status and the loss of the model 
    """
    model = build_pipeline(params)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_train)
    loss = log_loss(y_train, prob[:, 1])
    mlflow.log_metrics(
        {
            'train.log_loss': loss,
            'train.accuracy': accuracy_score(y_train, np.round(prob[:, 1]))
        }
    )
    return { 'status': STATUS_OK, 'loss': loss }