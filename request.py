import requests
import os
import glob
# 遠端
# url = "http://192.168.68.118:8000/process_image" 

# 本地
url = "http://localhost:8000/process_image"
# file_path = "C:\\Users\\11305064\\Documents\\VScode_Project\\VLM\\img\\lucas2.jpg"
directory_path = "C:\\Users\\11305064\\Documents\\VScode_Project\\VLM\\img_RAG\\"

# file_path = "your_image_path"

# Get a list of all .jpg files in the directory
file_paths = glob.glob(os.path.join(directory_path, "*.jpeg"))

for file_path in file_paths:
    with open(file_path, "rb") as f:
        files = {"file": f.read()}
        response = requests.post(url, files=files)
        print(response.json())