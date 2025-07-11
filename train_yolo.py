import mlflow
import torch
from ultralytics import YOLO
import os
import datetime

# Set up MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Use MLFLOW_EXPERIMENT_NAME if set, otherwise set it to "Monster Jam Detection"
if "MLFLOW_EXPERIMENT_NAME" in os.environ:
    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
else:
    experiment_name = "Monster Jam Detection"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

mlflow.set_experiment(experiment_name)

# Training configuration
data_config = "datasets/Monster-Jam-Detection-9/data.yaml"
model_name = "yolo11l.pt"
epochs = 100
img_size = 640

# Generate a unique run name using model, epoch count, and current timestamp
run_name = f"{model_name}-e{epochs}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Log GPU information (if available)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        mlflow.log_param("gpu_count", gpu_count)
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            mlflow.log_param(f"gpu_{i}_name", device_name)
            mlflow.log_param(f"gpu_{i}_memory_gb", f"{total_mem_gb:.2f}")
    else:
        mlflow.log_param("gpu_count", 0)

    # Train the YOLO model; Ultralytics logs automatically to the active MLflow run
    print("\nStarting YOLO training...")
    model = YOLO(model_name)
    model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        name=f"{model_name}-e{epochs}"
    )

    print("\nTraining complete. Results are available in MLflow.")
