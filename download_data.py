import os
import argparse
from roboflow import Roboflow

parser = argparse.ArgumentParser(description="Download a dataset from Roboflow.")
parser.add_argument("--api_key", type=str, default=None, help="Roboflow API key")
parser.add_argument("--workspace", type=str, default="cody-brownstein", help="Roboflow workspace name (default: cody-brownstein)")
parser.add_argument("--project", type=str, default="monster-jam-detection", help="Roboflow project name (default: monster-jam-detection)")
parser.add_argument("--version_number", type=int, default=9, help="Project version number (default: 9)")
parser.add_argument("--download_format", type=str, default="yolov8", help="Format to download (default: yolov8)")
args = parser.parse_args()

# --- Configuration Variables ---
API_KEY = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY must be provided as an argument or environment variable.")

WORKSPACE = args.workspace or os.environ.get("ROBOFLOW_WORKSPACE")
PROJECT = args.project or os.environ.get("ROBOFLOW_PROJECT")
VERSION_NUMBER = args.version_number or (int(os.environ.get("ROBOFLOW_VERSION_NUMBER")) if os.environ.get("ROBOFLOW_VERSION_NUMBER") else None)
DOWNLOAD_FORMAT = args.download_format or os.environ.get("ROBOFLOW_DOWNLOAD_FORMAT")
# --- End Configuration Variables ---

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION_NUMBER)
dataset = version.download(DOWNLOAD_FORMAT)

print(f"Dataset downloaded to: {dataset.location}")
