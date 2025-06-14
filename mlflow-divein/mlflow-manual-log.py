"""
LightGBM library requires OpenMP library (`libomp.dylib`) on Mac. To get this, install `libomp` using `Homebrew`.
Open the terminal and run: 
    `brew install libomp`
"""


"""                     Import libraries.                       """
import pandas as pd  # noqa: E402
import os  # noqa: E402
import joblib  # noqa: E402
from datetime import datetime as dt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import lightgbm as lgbm  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # noqa: E402


"""                     User defined variables.                       """
# Random seed.
random_seed = 14

# Mlflow.
# TODO - Make sure to start the mlflow server/ui on the specific port first.
mlflow_tracking_uri = 'http://localhost:8080'
mlflow_exp_name = 'mlflow-manual-log'  # Experiment name.
mlflow_run_name = 'lightgbm'

# Set the MLflow Tracking Server URI.
mlflow.set_tracking_uri(uri='http://localhost:8080')

# Miscellaneous.
folder_project = 'mlflow-divein'  # Project folder name.
folder_artifacts = 'artifacts'  # Folder to store artifacts locally.
folder_model = 'model'  # Folder to store the model.
folder_files = 'files'  # Folder to store files.
folder_plots = 'plots'  # Folder to store plots.

path_artifacts = os.path.join(os.getcwd(), folder_project, folder_artifacts)


"""                     Load and preprocess the data.                       """
# Load the breast cancer dataset.
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Basic checks on the dataset.
X.head()  # first few rows of the dataset.
X.shape  # shape of the dataset.
X.describe()  # dataset description.
X.isnull().sum()  # check for null values in the dataset.
X.isnull().sum().sum()  # Check for total null values in the dataset.

# Check the target variable.
y.head()  # first few rows of the target variable.
y.shape  # shape of the target variable.
y.describe()  # target variable description.
y.value_counts()  # distribution of the target variable.

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Create the folders to store artifacts, model, and files.
os.makedirs(os.path.join(path_artifacts, folder_model), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_files), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_plots), exist_ok=True)

# Ignoring any transformations (like, scaling) on the dataset for simplicity.

# Define parameters for the LightGBM model.
# https://lightgbm.readthedocs.io/en/latest/Parameters.html#parameters-format
params = {
    'objective': 'binary',
    'boosting': 'gbdt', # 'gbdt', 'rf', 'dart'
    'data_sample_strategy': 'goss', # bagging, goss
    'num_iterations': 100, # Number of boosting iterations.
    'learning_rate': 0.05, # Learning rate for the model.
    'num_leaves': 31, # Maximum number of leaves in one tree.
    'max_depth': -1, # Maximum depth of the tree. -1 means no limit.
    'feature_fraction': 0.8, # Fraction of features to be used for each boosting iteration.
    'early_stopping_round': 20, # Early stopping after 20 rounds if no improvement in validation loss.
    # 'bagging_fraction': 0.8, # Fraction of data to be used for each boosting iteration.
    # 'bagging_freq': 5, # Frequency of bagging. 0 means no bagging.
    # 'lambda_l1': 0.0, # L1 regularization term on weights.
    # 'lambda_l2': 0.0, # L2 regularization term on weights.
    'metric': 'auc', # auc, average_precision, binary_logloss
}


"""                     Start an MLflow run - 'manual logging'.                       """
# Create a LightGBM dataset.
train_dataset = lgbm.Dataset(X_train, label=y_train)
test_dataset = lgbm.Dataset(X_test, label=y_test, reference=train_dataset)

# Start an MLflow run with the defined experiment name and run name.
mlflow.set_experiment(mlflow_exp_name)
run_name = f"{mlflow_run_name}-{dt.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    # Log parameters to MLflow.
    mlflow.log_params(params)
    
    # Train and log the LightGBM model to MLflow.
    model_lgbm = lgbm.train(
        params=params,
        train_set=train_dataset,
        valid_sets=[test_dataset],
        # verbose_eval=False
    )
    signature = infer_signature(
        model_input=X_train,
        model_output=model_lgbm.predict(data=X_train)
    )
    # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html
    mlflow.lightgbm.log_model(
        lgb_model=model_lgbm, 
        name=folder_model,
        # name='model_lightgbm',
        signature=signature,
    )

    # Save the model locally.
    joblib.dump(
        value=model_lgbm,
        filename=os.path.join(path_artifacts, folder_model, 'model_lgbm.pkl')
    )

    # Predict on train and test datasets.
    y_train_pred = model_lgbm.predict(data=X_train)
    y_test_pred = model_lgbm.predict(data=X_test)

    y_train_pred_binary = (y_train_pred >= 0.5).astype(int)
    y_test_pred_binary = (y_test_pred >= 0.5).astype(int)

    """   Check the distribution of the predicted values.
    np.unique(y_test_pred_binary, return_counts=True)
    np.unique(y_train_pred_binary, return_counts=True)

    y_test.value_counts()
    y_train.value_counts()
    """

    # Calculate and log the model metrics to MLflow.
    dataset_types = ['train', 'test']
    y_trues = [y_train, y_test] # list of true labels for train and test.
    y_preds = [y_train_pred, y_test_pred] # list of predicted probabilities for train and test.
    y_preds_binary = [y_train_pred_binary, y_test_pred_binary] # list of predicted classes for train and test.

    for dataset, y_true, y_pred, y_pred_binary in zip(
        dataset_types, y_trues, y_preds, y_preds_binary
    ):
        model_metrics = {
            f"{dataset}_accuracy": accuracy_score(y_true=y_true, y_pred=y_pred_binary),
            f"{dataset}_precision": precision_score(y_true=y_true, y_pred=y_pred_binary),
            f"{dataset}_recall": recall_score(y_true=y_true, y_pred=y_pred_binary),
            f"{dataset}_f1_score": f1_score(y_true=y_true, y_pred=y_pred_binary),
            f"{dataset}_roc_auc": roc_auc_score(y_true=y_true, y_score=y_pred),
        }
        # print(model_metrics)
        mlflow.log_metrics(metrics=model_metrics)
    
    # Feature importance.
    df_feature_imp = pd.DataFrame(
        {
            'Feature': model_lgbm.feature_name(),
            'Importance_gain': model_lgbm.feature_importance(importance_type='gain'),
            'Importance_split': model_lgbm.feature_importance(importance_type='split'),
        }
    ).sort_values(by='Importance_gain', ascending=False).reset_index(drop=True)
    filename_tosave = 'feature_importance.csv'
    df_feature_imp.to_csv(
        path_or_buf=os.path.join(path_artifacts, folder_files, filename_tosave),
        index=False
    )
    mlflow.log_artifact(
        local_path=os.path.join(path_artifacts, folder_files, filename_tosave),
        artifact_path=folder_files
    )

    # Plot feature importance.
    fea_imp_types = ['split', 'gain']

    for type in fea_imp_types:
        ax = lgbm.plot_importance(
            model_lgbm,
            importance_type=type,
            max_num_features=10,
            figsize=(10, 8), # width=10, height=6
            title=f"Feature Importance ('{type}')",
        )
        filename_tosave = f"feature_importance_{type}.png"
        ax.figure.savefig(
            fname=os.path.join(path_artifacts, folder_plots, filename_tosave),
            bbox_inches='tight'
        )
        mlflow.log_artifact(
            local_path=os.path.join(path_artifacts, folder_plots, filename_tosave),
            artifact_path=folder_plots
        )
        # mlflow.log_figure(figure=ax.figure, artifact_file=os.path.join(folder_plots, filename_tosave)) # unable to save the image correctly. image gets cropped.
