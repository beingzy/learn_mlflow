import mlflow 

with mlflow.start_run():
	mlflow.log_metric("a", 1)