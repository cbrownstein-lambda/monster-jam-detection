import mlflow
import torch
from ultralytics import YOLO
import os
import datetime
import argparse

parser = argparse.ArgumentParser(description="Train YOLO model with MLflow tracking.")
parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5000", help="MLflow tracking URI (default: http://127.0.0.1:5000)")
parser.add_argument("--experiment_name", type=str, default="Monster Jam Detection", help="MLflow experiment name (default: Monster Jam Detection)")
parser.add_argument("--data_config", type=str, default="datasets/Monster-Jam-Detection-10/data.yaml", help="Path to data.yaml config file (default: datasets/Monster-Jam-Detection-10/data.yaml)")
parser.add_argument("--model_name", type=str, default="yolo11l.pt", help="YOLO model name (default: yolo11l.pt)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
parser.add_argument("--img_size", type=int, default=640, help="Image size for training (default: 640)")
args = parser.parse_args()

# Set up MLflow tracking URI
tracking_uri = args.tracking_uri
mlflow.set_tracking_uri(tracking_uri)

# Set up MLflow experiment name
experiment_name = args.experiment_name
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
mlflow.set_experiment(experiment_name)

# Training configuration
data_config = args.data_config
model_name = args.model_name
epochs = args.epochs
img_size = args.img_size

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
