import os
import shutil
import urllib.request
import tarfile
import argparse

def download_and_extract_gsc(data_dir):
    data_dir = os.path.join(data_dir, 'audio')
    os.makedirs(data_dir, exist_ok=True)
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")

    if not os.path.exists(os.path.join(data_dir, "validation_list.txt")):
        print(f"Downloading dataset into {data_dir}...")
        urllib.request.urlretrieve(url, archive_path)
        print("Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        os.remove(archive_path)
        print("Download and extraction done.")
    else:
        print("Dataset already appears to be downloaded and extracted.")

def move_files(src_folder, to_folder, list_file):
    with open(list_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            dirname = os.path.dirname(line)
            dest = os.path.join(to_folder, dirname)
            if not os.path.exists(dest):
                os.mkdir(dest)
            shutil.move(os.path.join(src_folder, line), dest)

def split_dataset(root_dir):
    audio_dir = os.path.join(root_dir, 'audio')
    val_list = os.path.join(audio_dir, "validation_list.txt")
    test_list = os.path.join(audio_dir, "testing_list.txt")
    valid_dir = os.path.join(root_dir, "valid")
    test_dir = os.path.join(root_dir, "test")
    train_dir = os.path.join(root_dir, "train")

    for directory in [valid_dir, test_dir, train_dir]:
        os.makedirs(directory, exist_ok=True)

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Moving validation files...")
    move_files(audio_dir, valid_dir, val_list)

    print("Moving test files...")
    move_files(audio_dir, test_dir, test_list)

    print("Moving remaining files to train folder...")
    os.rename(audio_dir, train_dir)

    print("Dataset successfully split into train/, valid/, test/.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GSC v2 dataset (download + split).')
    parser.add_argument('output_dir', type=str, help='Path to store and prepare the dataset.')
    args = parser.parse_args()

    download_and_extract_gsc(args.output_dir)
    split_dataset(args.output_dir)
