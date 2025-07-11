import os
from roboflow import Roboflow

# --- Configuration Variables ---
# Roboflow API key (should be set as an environment variable for security)
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

# Roboflow workspace name
WORKSPACE = "cody-brownstein"

# Roboflow project name
PROJECT = "monster-jam-detection"

# Project version number
# See the latest versions: https://universe.roboflow.com/cody-brownstein/monster-jam-detection
VERSION_NUMBER = 9

# Format to download (e.g., "yolov8")
DOWNLOAD_FORMAT = "yolov8"
# --- End Configuration Variables ---

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION_NUMBER)
dataset = version.download(DOWNLOAD_FORMAT)

print(f"Dataset downloaded to: {dataset.location}")
