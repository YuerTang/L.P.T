import os
import requests

def download_tinyshakespeare(data_dir="data"):
    """Downloads TinyShakespeare dataset if not already downloaded."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")

    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded dataset to {file_path}")
    else:
        print(f"Dataset already exists at {file_path}")

    return file_path

def load_tinyshakespeare(file_path):
    """Loads text data from TinyShakespeare dataset."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

if __name__ == "__main__":
    file_path = download_tinyshakespeare()
    text_data = load_tinyshakespeare(file_path)
    print(f"First 500 characters:\n{text_data[:500]}")
