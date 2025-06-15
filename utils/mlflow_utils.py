import mlflow
from typing import Dict, Any, Optional, Union
import tempfile
import os


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = False
) -> mlflow.ActiveRun:
    """Set up MLflow experiment and create a new run."""
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
    
    mlflow.set_experiment(experiment_name=experiment_name)
    return mlflow.start_run(run_name=run_name, nested=nested)

def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    for param_name, param_value in params.items():
        mlflow.log_param(key=param_name, value=param_value)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(key=metric_name, value=metric_value, step=step)

def log_model(model: Any, signature, model_name: str = 'model', conda_env: Optional[Dict] = None) -> None:
    """Log a model to MLflow."""
    mlflow.lightgbm.log_model(lgb_model=model, signature=signature, name=model_name, conda_env=conda_env)

def log_figure(figure, figure_name: str, artifact_path: str) -> None:
    """
    Log a figure to MLflow.
    Supports both matplotlib figures (saved as PNG) and plotly figures (saved as HTML).
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Handle Plotly figures (from newer Optuna versions)
        if hasattr(figure, 'write_html'):
            filepath = os.path.join(tmpdirname, f"{figure_name}.html")
            figure.write_html(filepath)
            mlflow.log_artifact(local_path=filepath, artifact_path=artifact_path)
        # Handle Matplotlib figures
        elif hasattr(figure, 'savefig'):
            filepath = os.path.join(tmpdirname, f"{figure_name}.png")
            figure.savefig(filepath)
            mlflow.log_artifact(local_path=filepath, artifact_path=artifact_path)
        else:
            print(f"Warning: Figure type {type(figure)} not supported for logging")

def log_optuna_study(study) -> None:
    """Log Optuna study details to MLflow."""
    # Log study attributes
    mlflow.log_param(key="n_trials", value=len(study.trials))
    
    # Log best trial values
    best_trial = study.best_trial
    mlflow.log_metrics(metrics={"best_value": best_trial.value})
    
    # Log best parameters with 'best_' prefix
    for key, value in best_trial.params.items():
        mlflow.log_param(key=f"best_{key}", value=value)
    
    # Log optimization plots
    try:
        import optuna.visualization as vis
        
        # Optimization history plot
        fig = vis.plot_optimization_history(study=study)
        log_figure(figure=fig, figure_name="optimization_history", artifact_path="optuna_plots")
        
        # Parameter importance
        fig = vis.plot_param_importances(study=study)
        log_figure(figure=fig, figure_name="param_importances", artifact_path="optuna_plots")
        
        # Add parallel coordinate plot if there are enough trials
        if len(study.trials) >= 2:
            fig = vis.plot_parallel_coordinate(study=study)
            log_figure(figure=fig, figure_name="parallel_coordinate", artifact_path="optuna_plots")
            
    except Exception as e:
        print(f"Warning: Could not log Optuna visualizations: {e}")