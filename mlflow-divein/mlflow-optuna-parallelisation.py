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
import optuna  # noqa: E402
import lightgbm as lgbm  # noqa: E402
import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)  # noqa: E402

# from utils.mlflow_utils import (setup_mlflow, log_params, log_metrics, log_model, log_optuna_study,)  # noqa: E402
import utils.mlflow_utils as mlfu  # noqa: E402


"""                     User defined variables.                       """
# Random seed.
random_seed = 14

# Mlflow.
mlflow_exp_name = "mlflow-optuna-parallelisation"  # Experiment name.
# mlflow_run_name = f"lgbm-{dt.now().strftime('%Y%m%d-%H%M%S')}" # Run name.

# TODO - Make sure to start the mlflow server/ui on the specific port first.
mlflow_tracking_uri = "http://localhost:8080"  # MLflow Tracking Server URI.
# mlflow.set_tracking_uri(uri='http://localhost:8080') # Set the MLflow Tracking Server URI.

# Miscellaneous.
folder_project = "mlflow-divein"  # Project folder name.
folder_artifacts = "artifacts"  # Folder to store artifacts locally.
folder_model = "model"  # Folder to store the model.
folder_files = "files"  # Folder to store files.
folder_plots = "plots"  # Folder to store plots.

path_artifacts = os.path.join(os.getcwd(), folder_project, folder_artifacts)


"""                     User defined funtions (for Optuna).                       """


# Model training function for LightGBM.
def train_lgbm_model(params_model, params_data):
    """Train model with given parameters and return metrics"""
    # Unpack the data parameters.
    X_train, y_train, X_test, y_test = (
        params_data["X_train"],
        params_data["y_train"],
        params_data["X_test"],
        params_data["y_test"],
    )

    # Create LightGBM datasets.
    ds_lgbm_train = lgbm.Dataset(data=X_train, label=y_train)
    ds_lgbm_test = lgbm.Dataset(data=X_test, label=y_test, reference=ds_lgbm_train)

    # Log the hyperparameters.
    mlfu.log_params(params=params_model)

    # Train the LightGBM model.
    model_lgbm = lgbm.train(
        params=params_model,
        train_set=ds_lgbm_train,
        valid_sets=[ds_lgbm_test],
    )

    # Predict on the test dataset.
    y_test_pred = model_lgbm.predict(data=X_test)

    # Calculate metrics.
    metrics = {
        "accuracy": accuracy_score(
            y_true=y_test, y_pred=(y_test_pred >= 0.5).astype(int)
        ),
        "precision": precision_score(
            y_true=y_test, y_pred=(y_test_pred >= 0.5).astype(int)
        ),
        "recall": recall_score(y_true=y_test, y_pred=(y_test_pred >= 0.5).astype(int)),
        "f1_score": f1_score(y_true=y_test, y_pred=(y_test_pred >= 0.5).astype(int)),
        "auc": roc_auc_score(y_true=y_test, y_score=y_test_pred),
        "average_precision": average_precision_score(
            y_true=y_test, y_score=y_test_pred
        ),
        "binary_logloss": log_loss(y_true=y_test, y_pred=y_test_pred),
    }

    # Get the model signature.
    signature = infer_signature(
        model_input=X_train.iloc[:1],
        model_output=model_lgbm.predict(data=X_train.iloc[:1]),
    )

    # Log the metrics.
    mlfu.log_metrics(metrics=metrics)

    # Log the model.
    mlfu.log_model(model=model_lgbm, signature=signature)

    return metrics


# Objective function for Optuna to optimize the LightGBM model.
def objective(
    trial,
    params_data,
    params_mlflow,
    mlflow_run_id=None,
    optimiser_metric="auc",
    run_parallel=False,  # Whether to run trials in parallel.
):
    """
    Objective function to optimize the LightGBM model using Optuna.
    """
    # Define hyperparameters to be optimized.
    params_lgbm = {
        "objective": "binary",
        "boosting": trial.suggest_categorical(
            name="boosting",
            choices=["gbdt", "rf", "dart"],
        ),
        "data_sample_strategy": trial.suggest_categorical(
            name="data_sample_strategy",
            choices=["bagging", "goss"],
        ),
        "num_iterations": trial.suggest_int(
            name="num_iterations",
            low=100,
            high=500,
            step=50,
        ),
        "learning_rate": trial.suggest_float(
            name="learning_rate",
            low=0.001,
            high=0.1,
            step=None,
            log=False,
        ),
        "num_leaves": trial.suggest_int(
            name="num_leaves",
            low=20,
            high=100,
            step=5,
        ),
        "max_depth": trial.suggest_int(
            name="max_depth",
            low=-1,
            high=20,
            step=1,
        ),
        "feature_fraction": trial.suggest_float(
            name="feature_fraction",
            low=0.5,
            high=1.0,
            step=None,
            log=False,
        ),
        # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        # 'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        # 'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        # 'metric': trial.suggest_categorical(
        #     name='metric',
        #     choices=['auc', 'average_precision', 'binary_logloss'],
        "metric": optimiser_metric,
    }

    # When running in parallel, we need to reset the tracking URI in each worker process
    if run_parallel and params_mlflow.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(uri=params_mlflow["mlflow_tracking_uri"])

    # Rest of the function remains the same
    if run_parallel:
        with mlfu.setup_mlflow(
            experiment_name=params_mlflow["mlflow_exp_name"],
            run_id=mlflow_run_id,
            nested=True,
        ):
            with mlfu.setup_mlflow(
                experiment_name=params_mlflow["mlflow_exp_name"],
                run_name=f"trial-{trial.number}",
                nested=True,
            ):
                metrics = train_lgbm_model(
                    params_model=params_lgbm, params_data=params_data
                )
    else:
        with mlfu.setup_mlflow(
            experiment_name=params_mlflow["mlflow_exp_name"],
            run_id=mlflow_run_id,
            run_name=f"trial-{trial.number}",
            nested=True,
        ):
            metrics = train_lgbm_model(
                params_model=params_lgbm, params_data=params_data
            )
            
    # Store all metrics in the trial's user_attrs for later retrieval
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, metric_value)
        
    return metrics[optimiser_metric]  # Return the metric to optimize (e.g., 'roc_auc').


# Optimization function to run the Optuna study.
def run_optimization(
    params_data,
    params_mlflow,
    direction="maximize",
    n_trials=20,
    optimiser_metric="auc",
    run_parallel=False,  # Whether to run trials in parallel.
):
    # Create a parent MLflow run for the entire optimization process.
    with mlfu.setup_mlflow(
        experiment_name=params_mlflow["mlflow_exp_name"],
        run_name=params_mlflow["mlflow_run_name"],
        tracking_uri=params_mlflow["mlflow_tracking_uri"],
        nested=False,
    ) as parent_run:
        # Get parent run ID for logging child runs.
        parent_run_id = parent_run.info.run_id

        # If running in parallel, set the number of jobs
        # For more stability, consider setting this to a specific number (like 4)
        # rather than -1, especially if you have many cores
        n_jobs = -1 if run_parallel else 1

        # Create and run the Optuna study.
        study = optuna.create_study(direction=direction)

        # Add a callback to show progress and help with synchronization
        def log_trial(study, trial):
            print(
                f"Trial {trial.number} finished with value: {trial.value} and params: {trial.params}"
            )

        try:
            study.optimize(
                func=lambda trial: objective(
                    trial=trial,
                    params_data=params_data,
                    params_mlflow=params_mlflow,
                    mlflow_run_id=parent_run_id,
                    optimiser_metric=optimiser_metric,
                    run_parallel=run_parallel,  # Pass the parallel flag to objective
                ),
                n_trials=n_trials,
                show_progress_bar=True,
                n_jobs=n_jobs,
                callbacks=[log_trial],
                timeout=600,  # Add a timeout to prevent hanging (10 minutes)
            )

            # Give a moment for all processes to finish and sync
            if run_parallel and len([t.state for t in study.trials])!=n_trials:
                print("Waiting for all trials to complete...")
                
                import time

                time.sleep(60)

            # # Log the Optuna study results to MLflow
            # mlfu.log_optuna_study(study=study)

            # # Check if we have a best trial before trying to access it
            # if study.trials and any(
            #     t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
            # ):
            #     print(f"\nBest trial: #{study.best_trial.number}")
            #     print(f"Value: {study.best_value}")
            #     print(f"Params:\n\t{study.best_params}")
            # else:
            #     print("No trials were successfully completed")

        except Exception as e:
            print(f"Error during optimization: {e}")

    return study


"""                     Load and preprocess the data.                       """
# Load the breast cancer dataset.
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Remove 'white-space' characters from column names.
X.columns = X.columns.str.replace(" ", "_", regex=True)

# Create the folders to store artifacts, model, and files.
os.makedirs(os.path.join(path_artifacts, folder_model), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_files), exist_ok=True)
os.makedirs(os.path.join(path_artifacts, folder_plots), exist_ok=True)


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


"""                     Hyperparameter optimisatiion using Optuna.                       """
# Define input parameters for the optimization process.
params_data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
}

mlflow_run_name = f"lgbm-auc-{dt.now().strftime('%Y%m%d-%H%M%S')}"  # Run name.

params_mlflow = {
    "mlflow_exp_name": mlflow_exp_name,
    "mlflow_run_name": mlflow_run_name,
    "mlflow_tracking_uri": mlflow_tracking_uri,
}

"""
'maximize' --> auc, 'average_precision'
'minimise' --> 'binary_logloss'
"""

# Run the optimization process with parallel execution - reduced n_jobs
study = run_optimization(
    params_data=params_data,
    params_mlflow=params_mlflow,
    direction="maximize",  # Direction to optimize the metric.
    n_trials=100,  # Number of trials to run.
    optimiser_metric="auc",  # Metric to optimize.
    run_parallel=True,  # Enable parallel execution
)

# Make sure to check if study has completed trials before accessing best_trial
if hasattr(study, "best_trial"):
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best trial value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")
else:
    print("No completed trials")

"""
params = {'boosting': 'rf', 'data_sample_strategy': 'bagging', 'num_iterations': 400, 'learning_rate': 0.049789231908604945, 'num_leaves': 20, 'max_depth': 11, 'feature_fraction': 0.9264411965784105}

print(params)

params['objective'] = 'binary'
params['metric'] = 'auc'

print(params)
"""

# Save the study results to a file.
study_file_path = os.path.join(
    path_artifacts,
    folder_files,
    f"optuna_study_{dt.now().strftime('%Y%m%d_%H%M%S')}.pkl",
)
joblib.dump(value=study, filename=study_file_path)

# # Save the best model to a file.
# best_model_file_path = os.path.join(
#     path_artifacts, folder_files, f"best_model_{dt.now().strftime('%Y%m%d_%H%M%S')}.pkl"
# )
# joblib.dump(value=study.best_trial.user_attrs['model'], filename=best_model_file_path)

# Print the best trial value and parameters.
