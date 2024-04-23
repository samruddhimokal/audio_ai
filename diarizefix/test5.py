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
import io
import srt
from datasets import load_dataset, Dataset

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
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

# Load the dataset from Hugging Face
dataset = load_dataset("AlabAI/youtube_wav", split="train", streaming=True)

# Process each audio file in the dataset
for i, data in enumerate(dataset):
    audio_bytes = data["audio"]["bytes"]
    
    # Convert audio bytes to AudioSegment
    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    
    # Save the audio segment to a temporary file
    temp_audio_path = "temp_audio.wav"
    audio_segment.export(temp_audio_path, format="wav")
    
    if args.stemming:
        # Isolate vocals from the rest of the audio
        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{temp_audio_path}" -o "temp_outputs"'
        )
        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            vocal_target = temp_audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(temp_audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = temp_audio_path
    
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
    sound = audio_segment.set_channels(1)
    temp_path = "temp_outputs"
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
    
    # Create a dictionary to store speaker-specific metadata
    speaker_metadata = {}
    
    # Generate the SRT file
    srt_data = ""
    for segment in ssm:
        start_time = segment["start"] / 1000
        end_time = segment["end"] / 1000
        speaker_name = f"Speaker {segment['speaker']}"
        transcript = segment["text"]
        
        srt_segment = srt.Subtitle(index=len(srt_data) + 1,
                                   start=srt.timedelta(seconds=start_time),
                                   end=srt.timedelta(seconds=end_time),
                                   content=f"{speaker_name}: {transcript}")
        srt_data += srt_segment.to_srt() + "\n"
    
    # Parse the SRT data
    srt_segments = list(srt.parse(srt_data))
    
    # Process each segment in the SRT data
    for segment in srt_segments:
        start_time = segment.start.total_seconds() * 1000
        end_time = segment.end.total_seconds() * 1000
        speaker_name, transcript = segment.content.split(": ", 1)
        
        # Extract the speaker ID from the speaker name
        speaker_id = int(speaker_name.split(" ")[-1])
        
        # Split the audio segment
        segment_audio = sound[start_time:end_time]
        segment_path = f"speaker_{speaker_id}_{segment.index:03d}.wav"
        segment_bytes = segment_audio.export(format="wav").read()
        
        # Store the metadata for each speaker
        if speaker_name not in speaker_metadata:
            speaker_metadata[speaker_name] = []
        speaker_metadata[speaker_name].append(f"speaker_{speaker_id}_{segment.index:03d}|{speaker_name}|{transcript}")
        
        # Upload the segment audio to the Hugging Face dataset
        dataset_dir = f"autodiarization/{i}/{speaker_id}"
        dataset_metadata = {"path": f"{dataset_dir}/{segment_path}", "speaker": speaker_name, "transcript": transcript}
        dataset = Dataset.from_dict({"audio": [{"bytes": segment_bytes, "path": segment_path}], "metadata": [dataset_metadata]})
        dataset.push_to_hub("AlabAI/audio_outputs", token="YOUR_ACCESS_TOKEN")
    
    # Write the metadata.csv file for each speaker
    for speaker_name, metadata in speaker_metadata.items():
        speaker_id = int(speaker_name.split(" ")[-1])
        dataset_dir = f"autodiarization/{i}/{speaker_id}"
        metadata_path = f"{dataset_dir}/metadata.csv"
        metadata_content = "\n".join(metadata)
        dataset = Dataset.from_dict({"metadata": [{"path": metadata_path, "content": metadata_content}]})
        dataset.push_to_hub("AlabAI/audio_outputs", token="YOUR_ACCESS_TOKEN")
    
    # Clean up temporary files
    cleanup(temp_path)
    os.remove(temp_audio_path)