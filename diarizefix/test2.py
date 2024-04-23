import argparse
import os
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import logging
import shutil

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation. This helps with long files that don't contain a lot of music.",
)
parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.",
)
parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)
parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
args = parser.parse_args()

if args.stemming:
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )
    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio

# Transcribe the audio file
if args.batch_size != 0:
    from transcription_helpers import transcribe_batched
    whisper_results, language = transcribe_batched(
        vocal_target,
        args.language,
        args.batch_size,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )
else:
    from transcription_helpers import transcribe
    whisper_results, language = transcribe(
        vocal_target,
        args.language,
        args.model_name,
        mtypes[args.device],
        args.suppress_numerals,
        args.device,
    )

if language in wav2vec2_langs:
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language, device=args.device
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, args.device
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=whisper_results[0].get("start"),
        final_timestamp=whisper_results[-1].get("end"),
    )
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    assert (
        args.batch_size == 0  # TODO: add a better check for word timestamps existence
    ), (
        f"Unsupported language: {language}, use --batch_size to 0"
        " to generate word timestamps using whisper directly and fix this error."
    )
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

# convert audio to mono for NeMo compatibility
sound = AudioSegment.from_file(vocal_target).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()
del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping
speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# Create the autodiarization directory structure
autodiarization_dir = "autodiarization"
os.makedirs(autodiarization_dir, exist_ok=True)

# Get the base name of the audio file
base_name = os.path.splitext(os.path.basename(args.audio))[0]

# Create a subdirectory for the current audio file
audio_dir = os.path.join(autodiarization_dir, base_name)
os.makedirs(audio_dir, exist_ok=True)

# Split the audio and create LJSpeech datasets for each speaker
for speaker_id in sorted(set(s[2] for s in speaker_ts)):
    speaker_dir = os.path.join(audio_dir, f"speaker_{speaker_id}")
    os.makedirs(speaker_dir, exist_ok=True)

    speaker_segments = [s for s in ssm if s["speaker"] == speaker_id]

    metadata = []
    for i, segment in enumerate(speaker_segments, start=1):
        start_time = segment["start"] / 1000
        end_time = segment["end"] / 1000
        transcript = " ".join(w["word"] for w in segment["words"])

        # Split the audio segment
        segment_audio = sound[start_time * 1000 : end_time * 1000]
        segment_path = os.path.join(speaker_dir, f"speaker_{speaker_id}_{i:03d}.wav")
        segment_audio.export(segment_path, format="wav")

        metadata.append(f"speaker_{speaker_id}_{i:03d}|speaker_{speaker_id}|{transcript}")

    # Write the metadata.csv file for the speaker
    with open(os.path.join(speaker_dir, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))

# Write the full transcript and SRT files
with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8") as srt:
    write_srt(ssm, srt)

# Clean up temporary files
cleanup(temp_path)
