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
import time  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss  # noqa: E402
from utils.plot_utils import plot_correlation_matrix  # noqa: E402
from utils import mlflow_utils as mlfu  # noqa: E402
from  utils import model_eval_utils as mevalu  # noqa: E402
from utils import data_preprocess_utils as dpu  # noqa: E402


"""                     User defined variables.                       """
# Random seed.
random_seed = 14

# Mlflow.
# TODO - Make sure to start the mlflow server/ui on the specific port first.
mlflow_tracking_uri = 'http://localhost:8080' # MLflow Tracking Server URI.
mlflow_exp_name = 'mlflow-optuna-parallel'  # Experiment name.
# mlflow.set_tracking_uri(uri='http://localhost:8080') # Set the MLflow Tracking Server URI.

# Miscellaneous.
folder_project = 'mlflow-divein'  # Project folder name.
folder_artifacts = 'artifacts'  # Folder to store artifacts locally.
folder_model = 'model'  # Folder to store the model.
folder_files = 'files'  # Folder to store files.
folder_plots = 'plots'  # Folder to store plots.

path_artifacts = os.path.join(os.getcwd(), folder_project, folder_artifacts)


"""                     User defined funtions (for Optuna).                       """
# Model training function for LightGBM.
def train_lgbm_model(params_model, params_data, best_model=False):
    """Train model with given parameters and return metrics"""

    # Unpack the data parameters.
    X_train = params_data['X_train']
    y_train = params_data['y_train']
    X_test = params_data['X_test']
    y_test = params_data['y_test']

    # Define the threshold for binary classification.
    threshold = params_data['threshold'] if 'threshold' in params_data else 0.5

    # Get the LightGBM dataset.
    ds_train = lgbm.Dataset(data=X_train, label=y_train)
    ds_val = lgbm.Dataset(data=X_test, label=y_test, reference=ds_train)

    # Train the LightGBM model.
    model_lgbm = lgbm.train(
        params=params_model,
        train_set=ds_train,
        valid_sets=[ds_val],
    )

    # Predict on the test dataset.
    y_test_pred = model_lgbm.predict(data=X_test)

    # Calculate metrics.
    metrics = {
        'accuracy': accuracy_score(y_true=y_test, y_pred=(y_test_pred >= threshold).astype(int)),
        'precision': precision_score(y_true=y_test, y_pred=(y_test_pred >= threshold).astype(int)),
        'recall': recall_score(y_true=y_test, y_pred=(y_test_pred >= threshold).astype(int)),
        'f1_score': f1_score(y_true=y_test, y_pred=(y_test_pred >= threshold).astype(int)),
        'auc': roc_auc_score(y_true=y_test, y_score=y_test_pred),
        'average_precision': average_precision_score(y_true=y_test, y_score=y_test_pred),
        'binary_logloss': log_loss(y_true=y_test, y_pred=y_test_pred),
    }

    # Get the model signature.
    signature = infer_signature(
        model_input=X_train.iloc[:1], 
        model_output=model_lgbm.predict(data=X_train.iloc[:1])
    )

    # Log advanced model evaluation metrics, if best_model is True.
    if best_model:
        # Rename the attributes.
        params_model = {f"best_{k}": v for k, v in params_model.items()}
        metrics = {f"best_{k}": v for k, v in metrics.items()}
        
        # Log advanced model evaluation metrics.
        # TODO - Implement the advanced model evaluation metrics logging.
        mevalu.model_eval_binary_classification(
            model=model_lgbm, 
            params_data=params_data,
            threshold=threshold, 
        )

    # Define the model artifact path.
    model_art_path = 'best_model' if best_model else 'model'

    # Log model parameters, metrics and model in MLflow.
    mlfu.log_params(params=params_model)
    mlfu.log_metrics(metrics=metrics)
    mlfu.log_model(model=model_lgbm, signature=signature, model_art_path=model_art_path)
    
    return metrics

# Objective function for Optuna to optimize the LightGBM model.
def objective(
    trial, 
    params_model_static,
    params_data,
    params_mlflow,
    optimiser_metric, 
): 
    """
    Objective function to optimize the LightGBM model using Optuna.
    """
    # Define hyperparameters to be optimized.
    params_lgbm = {
        # 'objective': 'binary',
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
        # 'metric': trial.suggest_categorical(
        #     name='metric', 
        #     choices=['auc', 'average_precision', 'binary_logloss'],
    }
    
    # Combine static parameters with the trial parameters.
    params_lgbm.update(params_model_static)

    # Create a nested MLflow run for this trial.
    with mlfu.setup_mlflow(
        parent_run_id=params_mlflow['parent_run_id'],
        run_name=f"trial-{trial.number}",
        nested=True
    ):
        # Train the model and get the metrics.
        metrics = train_lgbm_model(
            params_model=params_lgbm, 
            params_data=params_data,
        )
  
    return metrics[optimiser_metric]  # Return the metric to optimize (e.g., 'auc').

# Optimization function to run the Optuna study.
def run_optimization(
    params_model_static,
    params_data,
    params_mlflow,
    params_study,
):  
    # Create a parent MLflow run for the entire optimization process.
    with mlfu.setup_mlflow(
        tracking_uri=params_mlflow['mlflow_tracking_uri'],
        experiment_name=params_mlflow['mlflow_exp_name'], 
        run_name=params_mlflow['mlflow_run_name'],
        nested=False
    ) as parent_run:
        # Define paameters for parallel or sequential runs.
        n_jobs = -1 if params_study['run_parallel'] else 1  # Use all available CPU cores for parallel execution, or single-threaded execution for simplicity.
        params_mlflow['parent_run_id'] = parent_run.info.run_id if params_study['run_parallel'] else None # Get the parent run ID for nested runs in parallel.

        # Create and run the Optuna study.
        study = optuna.create_study(
            direction=params_study['direction'],
            study_name=params_study['name'],  # Name of the study.
        )
        study.optimize(
            func=lambda trial: objective(
                trial=trial, 
                params_model_static=params_model_static,
                params_data=params_data, 
                params_mlflow=params_mlflow,
                optimiser_metric=params_study['optimiser_metric'],
            ),
            n_trials=params_study['n_trials'], # Number of trials to run.
            timeout=None,  # No timeout for the optimization.
            n_jobs=n_jobs,  # Number of parallel jobs to run.  
            catch=(Exception,),  # Catch all exceptions during the optimization.
            gc_after_trial=True,  # Garbage collect after each trial to free memory. 
            show_progress_bar=True, # Show progress bar for the optimization.
        )

        # Check all the trials have finished or not, when running in parallel.
        if params_study['run_parallel']:
            # Get the number of finished trials.
            trial_count_finished = sum(
                optuna.trial.TrialState.is_finished(t.state) for t in study.trials
            )

            # Wait until all trials are finished.
            while trial_count_finished < params_study['n_trials']:
                print(f"Waiting for {(params_study['n_trials'] - trial_count_finished)} trials to finish...")
                
                trial_count_finished = sum(
                    optuna.trial.TrialState.is_finished(t.state) for t in study.trials
                )
                # Wait for a short period before checking again.
                time.sleep(30)

        # Log the Optuna study results to MLflow.
        flag_study = mlfu.log_optuna_study(study=study, params_study=params_study)

        """
        - Add the 'objective' and 'metric' to the best parameters. This is necessary for LightGBM to work correctly (build and predict). Without the objective function, predict function returns raw scores instead of probabilities, which can be less then 0 and/or greater than 1.
        - Since these aren't being tuned/optimised, they aren't captured in the 'study.best_params'. 
        
        """
        # If the study was logged successfully, log the best model.
        if flag_study:
            print("Logging the best model to MLflow...\n")
            
            # Get the model parameters for the best trial.
            params_model_best = params_model_static.copy()
            params_model_best.update(study.best_params)

            # Build the best model.
            _ = train_lgbm_model(
                params_model=params_model_best, 
                params_data=params_data,
                best_model=True,  # Log advanced model metrics for best model.
            )

    return study


"""                     Load and preprocess the data.                       """
# Load the breast cancer dataset.
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Remove 'white-space' characters from column names.
X.columns = X.columns.str.replace(' ', '_', regex=True)

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
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=random_seed
# )
X_train, X_val, X_test, y_train, y_val, y_test = dpu.split_data(
    X=X,
    y=y,
    split_test=0.2,  # 20% of the data for testing.
    split_val=0.1,  # 10% of the data for validation.
    random_state=random_seed,  # Random seed for reproducibility.
)

# Ignoring any transformations (like, scaling) on the dataset for simplicity.


"""                     Hyperparameter optimisatiion using Optuna.                       """
# Define the optimiser for study.
"""
'maximize' --> 'auc', 'average_precision'
'minimise' --> 'binary_logloss'
"""
optimiser = {
    'name': 'average_precision', # Name of the metric to optimize.
    'direction': 'maximize', # Direction to optimize the metric.
    'alias': 'avg-precision', # Alias for the metric.
}

# Define the study parameters.
params_study = {
    'name': f"lgbm-{optimiser['alias']}-study",  # Name of the study.
    'n_trials': 5,  # Number of trials to run.
    'direction': optimiser['direction'],  # Direction to optimize the metric.
    'optimiser_metric': optimiser['name'],  # Metric to optimize.
    'run_parallel': False,  # Set to True for parallel execution, False for sequential.
}

# Define the static parameters for the LightGBM model.
params_model_static = {
    'objective': 'binary',  # Objective function for LightGBM.
    'metric': optimiser['name'],  # Metric to optimize.
    'num_threads': os.cpu_count(),  # Use all available CPU cores.
}

# Define the data parameters.
params_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'threshold': 0.5,  # Threshold for binary classification.
}

# Define the MLflow parameters.
params_mlflow = {
    'mlflow_exp_name': mlflow_exp_name,
    'mlflow_tracking_uri': mlflow_tracking_uri,
    'mlflow_run_name': f"lgbm-{optimiser['alias']}-{dt.now().strftime('%Y%m%d-%H%M%S')}",
    'parent_run_id': None,  # This will be set during the optimization process.
}

# Run the optimization process.
study = run_optimization(
    params_model_static=params_model_static,
    params_data=params_data,
    params_mlflow=params_mlflow,
    params_study=params_study
)

# Save the study results to a file.
study_file_path = os.path.join(
    path_artifacts, folder_files, f"{params_mlflow['mlflow_run_name']}.pkl"
)
joblib.dump(value=study, filename=study_file_path)

# Print the best trial value and parameters.
print(f"Best trial number: {study.best_trial.number}")
print(f"Best trial value: {study.best_value}")
print(f"Best parameters: {study.best_params}")
