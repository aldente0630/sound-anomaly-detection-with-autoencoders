import os
import pickle
import zipfile
from urllib import request


def dump_pickle(file_path, obj):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def build_files_list(root_dir, normal_dir="normal", abnormal_dir="abnormal"):
    normal_files = []
    abnormal_files = []

    for root, _, files in os.walk(top=os.path.join(root_dir)):
        for name in files:
            current_dir_type = root.split("/")[-1]
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))

    return normal_files, abnormal_files


def download_and_unzip_file(dir_path, file_name):
    file_path = os.path.join(dir_path, f"{file_name}.zip")

    if os.path.exists(file_path):
        print("Sound files found, no need to download them again.")

    else:
        print(
            f"Downloading and unzipping the file, {file_name}.zip from the MIMII dataset website..."
        )
        url = f"https://zenodo.org/record/3384388/files/{file_name}.zip?download=1"
        request.urlretrieve(url, file_path)

        if file_name.startswith("-"):
            database = "min" + "".join(file_name.split("_")[:-1])[1:]
        else:
            database = "".join(file_name.split("_")[:-1])

        zip_file = zipfile.ZipFile(file_path)
        zip_file.extractall(os.path.join(dir_path, database))
        zip_file.close()
        print("Done.")
