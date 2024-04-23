from __future__ import unicode_literals
import yt_dlp
import csv
import os
from huggingface_hub import HfApi

def download_and_convert(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '64',
        }],
        'outtmpl': '%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        output_file = filename.replace('.webm', '.wav')

    return output_file

def upload_to_huggingface(file_path):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="AlabAI/youtube_wav_files",
        repo_type="dataset",
    )

def delete_local_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")
    except OSError as e:
        print(f"Error deleting local file: {file_path}")
        print(f"Error details: {e}")

# Read URLs from youtube.csv
with open('youtube.csv', 'r') as file:
    csv_reader = csv.reader(file)
    urls = [row[0] for row in csv_reader]

# Process each URL
for url in urls:
    output_file = download_and_convert(url)
    if output_file:
        upload_to_huggingface(output_file)
        print(f"Uploaded {output_file} to Hugging Face.")
        delete_local_file(output_file)
    else:
        print(f"Skipping upload for {url} due to conversion failure.")