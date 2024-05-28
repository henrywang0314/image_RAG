import requests
import os
import glob
# 遠端
# url = "http://192.168.68.142:8000/process_image" 

# 本地
url = "http://localhost:8000/process_image"
# file_path = "C:\\Users\\11305064\\Documents\\VScode_Project\\VLM\\img\\lucas2.jpg"
# directory_path = "C:\\Users\\11305064\\Documents\\VScode_Project\\VLM\\img_RAG\\"

# current directory
directory_path = os.getcwd()
directory_path = os.path.join(directory_path, "Img_RAG")

# file_path = "your_image_path"
print(directory_path)
# Get a list of all .jpg files in the directory
# /home/apteam/Henry/image_RAG/Img_RAG/IMG_0590.jpeg
file_paths = glob.glob(os.path.join(directory_path, "*.jpeg"))
print(file_paths)
input("Press Enter to continue...")
for file_path in file_paths:
    with open(file_path, "rb") as f:
        files = {"file": f.read()}
        response = requests.post(url, files=files)
        print(response.json())