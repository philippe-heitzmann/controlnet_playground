import os

def create_dir_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
