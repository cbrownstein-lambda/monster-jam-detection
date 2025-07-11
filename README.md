# Monster Jam Detection

## Project Overview
This repository provides code and configuration for training a [YOLO object detection model](https://docs.ultralytics.com/models/yolo11/) to identify die-cast toy [Monster Jam trucks](https://www.monsterjam.com/trucks/) using a [Roboflow](https://roboflow.com/) dataset and [MLflow](https://mlflow.org/) for experiment tracking.

### What is YOLO?
YOLO (**Y**ou **O**nly **L**ook **O**nce) is a cutting-edge object detection algorithm known for its incredible speed and efficiency. It examines an entire image in a single pass to identify all objects, making it ideal for real-time applications. 🚀

---

### The Dataset
This project is trained on a [custom dataset](https://universe.roboflow.com/cody-brownstein/monster-jam-detection) of my kid's **die-cast toy** Monster Jam trucks, which he lovingly leaves scattered all over our home. The images are managed on Roboflow and include augmentations—like rotations—to help the model recognize the toy trucks from any angle. The dataset is also [mirrored on Hugging Face](https://huggingface.co/datasets/cbrownstein-lambda/monster-jam). 🤗

---

### Project Goal
The primary goal is to train a model that can identify the various die-cast toy Monster Jam trucks with at least 90% accuracy.

## Getting Started
Clone the repository:
```bash
git clone https://github.com/cbrownstein-lambda/monster-jam-detection.git
cd monster-jam-detection
```

## Prerequisites
- Python 3.8+
- Docker (for MLflow server)
- Roboflow account and API key
- GPU recommended for training
- Recommended: Use a Python virtual environment (venv) to avoid package conflicts.
- Required Python modules (install with pip):
  - `mlflow`
  - `torch`
  - `ultralytics`
  - `roboflow`

### Set up a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install all required modules:
```bash
pip install -U mlflow torch ultralytics roboflow
```

## 1. Start MLflow Tracking Server
MLflow is used to track experiments and results. The project includes a `mlflow.compose.yaml` for easy setup with Docker Compose.

> **Note for Lambda On-Demand Cloud users:**
> The `ubuntu` user may need to be added to the `docker` group to run Docker commands without `sudo`:
> ```bash
> sudo usermod -aG docker ubuntu
> # Log out and back in for group changes to take effect
> ```
> Alternatively, you can prefix Docker commands with `sudo`:
> ```bash
> sudo docker compose -f mlflow.compose.yaml up -d
> ```

```bash
# Start MLflow server (from project root)
docker compose -f mlflow.compose.yaml up -d
```
- MLflow UI will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Data is stored in a Docker volume (`mlflow_backend`)

## 2. Download the Dataset from Roboflow
Use `download_data.py` to fetch the dataset. You need a Roboflow API key. You can pass it as an argument or set it as an environment variable.

```bash
# Set your Roboflow API key (recommended)
export ROBOFLOW_API_KEY=your_api_key_here

# Download the dataset (default settings)
python download_data.py

# Custom options:
python download_data.py --api_key your_api_key --workspace cody-brownstein --project monster-jam-detection --version_number 10 --download_format yolov8
```
**Defaults:**
- Workspace: `cody-brownstein`
- Project: `monster-jam-detection`
- Version: `10`
- Format: `yolov8`

The dataset will be downloaded to a folder printed in the output.

> ‼️ **Note:** ‼️ Downloaded datasets are expected to be placed in the `datasets/` directory. Make sure your data config path (e.g., `datasets/Monster-Jam-Detection-9/data.yaml`) matches the location of your downloaded dataset.
>
> You can also download the datasets directly from [Roboflow Universe](https://universe.roboflow.com/cody-brownstein/monster-jam-detection).

## 3. Train the YOLO Model
Use `train_yolo.py` to start training. MLflow will automatically log parameters and results.

> **Important:** Ensure MLflow logging is enabled in YOLO settings. In Ultralytics YOLO, set `mlflow=True` in your training command or configuration if it's not already enabled. See the [Ultralytics MLflow integration docs](https://docs.ultralytics.com/integrations/mlflow/) for details.

```bash
# Train with default settings
python train_yolo.py

# Custom options:
python train_yolo.py \
  --tracking_uri http://127.0.0.1:5000 \
  --experiment_name "Monster Jam Detection" \
  --data_config datasets/Monster-Jam-Detection-9/data.yaml \
  --model_name yolo11l.pt \
  --epochs 100 \
  --img_size 640
```
**Defaults:**
- Tracking URI: `http://127.0.0.1:5000`
- Experiment Name: `Monster Jam Detection`
- Data Config: `datasets/Monster-Jam-Detection-9/data.yaml`
- Model Name: `yolo11l.pt`
- Epochs: `100`
- Image Size: `640`

## 4. View Results
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to view experiment runs, metrics, and artifacts.

## Environment Variables
You can set the following environment variables to override defaults:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION_NUMBER`
- `ROBOFLOW_DOWNLOAD_FORMAT`
- `MLFLOW_EXPERIMENT_NAME`

---
For questions or issues, please open an issue on GitHub.
