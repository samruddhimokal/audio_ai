import os
import tempfile
from pytube import YouTube
import re
import shutil
import csv

# Set the directory where the WAV files will be saved
output_directory = "youtube"
repo_id = "samrm/youtube_wav_test"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the YouTube links from the CSV file
with open("youtube.csv", "r") as file:
    reader = csv.reader(file)
    video_urls = [row[0] for row in reader]

def process_video(url):
    # Download video using pytube
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True, abr="128kbps").first()
    
    # Clean the video title for use as a file name
    cleaned_title = re.sub(r'[^\w\-_\. ]', '_', yt.title)
    
    # Create a temporary directory for the downloaded audio
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, "audio.wav")
        print(f"Downloading audio: {yt.title}")
        stream.download(output_path=temp_dir, filename="audio.wav")
    
        # Move the downloaded WAV file to the output directory
        wav_file_path = os.path.join(output_directory, cleaned_title + ".wav")
        print(f"Moving WAV file: {yt.title}")
        shutil.move(temp_audio_path, wav_file_path)
    
    # Push the WAV file to the dataset repository
    print(f"Pushing to dataset repository: {cleaned_title}")
    os.system(f'huggingface-cli upload --repo-type dataset {repo_id} "{wav_file_path}"')
    
    # Delete the saved WAV file
    os.unlink(wav_file_path)
    
    print(f"Processed and pushed: {cleaned_title}")

# Process the videos sequentially
total_videos = len(video_urls)
for index, url in enumerate(video_urls, start=1):
    print(f"Processing video {index}/{total_videos}")
    process_video(url)

print("All videos processed and pushed to the dataset repository.")