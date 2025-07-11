import os
import argparse
from roboflow import Roboflow

parser = argparse.ArgumentParser(description="Download a dataset from Roboflow.")
parser.add_argument("--api_key", type=str, default=None, help="Roboflow API key")
parser.add_argument("--workspace", type=str, default=None, help="Roboflow workspace name")
parser.add_argument("--project", type=str, default=None, help="Roboflow project name")
parser.add_argument("--version_number", type=int, default=None, help="Project version number")
parser.add_argument("--download_format", type=str, default=None, help="Format to download (e.g., yolov8)")
args = parser.parse_args()

# --- Configuration Variables ---
API_KEY = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY must be provided as an argument or environment variable.")

WORKSPACE = args.workspace or os.environ.get("ROBOFLOW_WORKSPACE", "cody-brownstein")
PROJECT = args.project or os.environ.get("ROBOFLOW_PROJECT", "monster-jam-detection")
VERSION_NUMBER = args.version_number or int(os.environ.get("ROBOFLOW_VERSION_NUMBER", 9))
DOWNLOAD_FORMAT = args.download_format or os.environ.get("ROBOFLOW_DOWNLOAD_FORMAT", "yolov8")
# --- End Configuration Variables ---

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION_NUMBER)
dataset = version.download(DOWNLOAD_FORMAT)

print(f"Dataset downloaded to: {dataset.location}")
