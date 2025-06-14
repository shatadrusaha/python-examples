"""
LightGBM library requires OpenMP library (`libomp.dylib`) on Mac. To get this, install `libomp` using `Homebrew`.
Open the terminal and run: 
    `brew install libomp`

Documentation:
    https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs
"""


"""                     Import libraries.                       """
import pandas as pd  # noqa: E402
import os  # noqa: E402
import joblib  # noqa: E402
from datetime import datetime as dt  # noqa: E402
import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
import optuna  # noqa: E402
import lightgbm as lgbm  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # noqa: E402
from utils.generic_utils import plot_correlation_matrix  # noqa: E402


"""                     User defined variables.                       """
# Random seed.
random_seed = 14

# Mlflow.
mlflow_exp_name = 'mlflow-optuna'  # Experiment name.
mlflow_run_name = 'lightgbm' # Run name.

# TODO - Make sure to start the mlflow server/ui on the specific port first.
mlflow_tracking_uri = 'http://localhost:8080' # MLflow Tracking Server URI.
mlflow.set_tracking_uri(uri='http://localhost:8080') # Set the MLflow Tracking Server URI.

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

# Create the folders to store artifacts, model, and files.
os.makedirs(os.path.join(path_artifacts, folder_model), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_files), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_plots), exist_ok=True)

# Get the correlation matrix and plot for the dataset.
corr_w_target = plot_correlation_matrix(
    df=pd.concat(objs=[X, y], axis=1),  # Concatenate X and y to get the target variable in the correlation matrix.
    col_target='target',
    figsize=(10, 8),  # width=10, height=6
    folder_tosave_plot=os.path.join(
        path_artifacts, folder_plots
    ),
)

corr_wo_target = plot_correlation_matrix(
    df=pd.concat(objs=[X, y], axis=1),  # Concatenate X and y to get the target variable in the correlation matrix.
    col_target=None,
    figsize=(20, 20),  # width=10, height=6
    folder_tosave_plot=os.path.join(
        path_artifacts, folder_plots
    ),
)

"""
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
"""

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Ignoring any transformations (like, scaling) on the dataset for simplicity.

# Define an objective function for Optuna to optimize the LightGBM model.
def objective(trial):
    """
    Objective function to optimize the LightGBM model using Optuna.
    """
    # Define hyperparameters to be optimized.
    params = {
        'objective': 'binary',
        'boosting': trial.suggest_categorical(
            name='boosting', 
            choices=['gbdt', 'rf', 'dart'],
        ),
        'data_sample_strategy': trial.suggest_categorical(
            name='data_sample_strategy', 
            choices=['bagging', 'goss'],
        ),
        'num_iterations': trial.suggest_int(
            name='num_iterations', 
            low=100, 
            high=500,
            step=50,
        ),
        'learning_rate': trial.suggest_float(
            name='learning_rate', 
            low=0.001, 
            high=0.1,
            step=None,
            log=False,
            ),
        'num_leaves': trial.suggest_int(
            name='num_leaves', 
            low=20, 
            high=100,
            step=5,
        ),
        'max_depth': trial.suggest_int(
            name='max_depth', 
            low=-1, 
            high=20,
            step=1,
        ),
        'feature_fraction': trial.suggest_float(
            name='feature_fraction', 
            low=0.5, 
            high=1.0,
            step=None,
            log=False,
        ),
        # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        # 'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        # 'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'metric': trial.suggest_categorical(
            name='metric', 
            choices=['auc', 'average_precision', 'binary_logloss'],
        ),
    }

    # # Create a LightGBM dataset.
    # train_dataset = lgbm.Dataset(X_train, label=y_train)
    # test_dataset = lgbm.Dataset(X_test, label=y_test, reference=train_dataset)

    # # Train the LightGBM model.
    # model_lgbm = lgbm.train(
    #     params=params,
    #     train_set=train_dataset,
    #     valid_sets=[test_dataset],
    #     verbose_eval=False
    # )

    # # Predict on the test dataset.
    # y_test_pred = model_lgbm.predict(data=X_test)

    # # Calculate the ROC AUC score.
    # roc_auc = roc_auc_score(y_true=y_test, y_score=y_test_pred)

    return roc_auc

# Define a train fun



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
