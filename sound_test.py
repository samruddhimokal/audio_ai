import os
import wave
import struct
import soundfile as sf

# Set the path to the audio file
audio_file = "Aaron Smith-Levin_ Scientology _ Lex Fridman Podcast _361.wav"

# Check if the file exists
if not os.path.isfile(audio_file):
    print(f"File not found: {audio_file}")
    exit(1)

# Get the file size
file_size = os.path.getsize(audio_file)
print(f"File size: {file_size} bytes")

# Try to open the file using soundfile
try:
    data, sample_rate = sf.read(audio_file)
    print(f"File successfully opened using soundfile")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of channels: {data.shape[1]}")
    print(f"Duration: {len(data) / sample_rate:.2f} seconds")
except Exception as e:
    print(f"Error opening file using soundfile: {str(e)}")

# Try to open the file using wave module
try:
    with wave.open(audio_file, "rb") as wav_file:
        print(f"File successfully opened using wave module")
        print(f"Number of channels: {wav_file.getnchannels()}")
        print(f"Sample width: {wav_file.getsampwidth()} bytes")
        print(f"Frame rate: {wav_file.getframerate()} Hz")
        print(f"Number of frames: {wav_file.getnframes()}")
        print(f"Compression type: {wav_file.getcomptype()}")
        print(f"Compression name: {wav_file.getcompname()}")
except Exception as e:
    print(f"Error opening file using wave module: {str(e)}")

# Read the file header
try:
    with open(audio_file, "rb") as file:
        header = file.read(44)
        riff, size, wave_fmt = struct.unpack("<4sI4s", header[:12])
        print(f"RIFF header: {riff.decode('ascii')}")
        print(f"File size: {size + 8} bytes")
        print(f"WAVE format: {wave_fmt.decode('ascii')}")
except Exception as e:
    print(f"Error reading file header: {str(e)}")