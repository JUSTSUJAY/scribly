import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import whisper
import shutil
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_ffmpeg_path():
    """Get the full path to ffmpeg executable"""
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        # Default Windows installation path for winget
        ffmpeg_path = r"E:\kalvium\dec_sprint\knet_mock\ffmpeg-2024-11-28-git-bc991ca048-essentials_build\bin\ffmpeg.exe"
    return ffmpeg_path

def download_video_audio(video_url: str, output_path: Path) -> Path:
    """Downloads the audio track from a video using yt-dlp."""
    logging.info(f"Downloading audio for: {video_url}")
    
    video_id = video_url.split("v=")[-1]
    audio_filepath = output_path / f"{video_id}.m4a"
    
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "m4a",
        "--audio-quality", "0",
        "--no-playlist",
        "--ignore-errors",
        "-o", str(audio_filepath),
        video_url
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        # Convert webm to m4a if needed
        if not audio_filepath.exists():
            webm_file = output_path / f"{video_id}.webm"
            if webm_file.exists():
                ffmpeg_command = [
                    "ffmpeg", "-i", str(webm_file),
                    "-c:a", "aac", str(audio_filepath)
                ]
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                webm_file.unlink()  # Remove webm file after conversion
        return audio_filepath if audio_filepath.exists() else None
    except subprocess.CalledProcessError:
        return None


def transcribe_audio(audio_file: Path, model: whisper.Whisper) -> str:
    """Transcribes audio using Whisper."""
    if not audio_file or not audio_file.exists():
        return ""

    try:
        logging.info(f"Transcribing {audio_file}")
        result = model.transcribe(str(audio_file.absolute()))
        return result["text"]
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        return ""

def process_video(video_url: str, output_dir: Path, model: whisper.Whisper) -> bool:
    """Process a single video: download and transcribe."""
    video_id = video_url.split("v=")[-1]
    video_dir = output_dir / video_id
    video_dir.mkdir(exist_ok=True, parents=True)
    
    audio_filepath = download_video_audio(video_url, video_dir)
    if audio_filepath and audio_filepath.exists():
        transcript_text = transcribe_audio(audio_filepath, model)
        if transcript_text:
            transcript_file = video_dir / f"{video_id}_transcript.txt"
            transcript_file.write_text(transcript_text, encoding="utf-8")
            logging.info(f"Successfully transcribed: {video_id}")
            return True
    return False

def bulk_transcribe_videos(video_urls: list, model_name: str = "base", 
                          workers: int = 4, output_dir: str = "transcripts"):
    """Bulk transcribe videos with progress tracking."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load Whisper model explicitly
    logging.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    logging.info("Model loaded successfully")
    
    successful = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_video, url, output_path, model) 
                  for url in video_urls]
        
        for future in tqdm(futures, desc="Transcribing videos"):
            try:
                if future.result():
                    successful += 1
            except Exception as e:
                logging.error(f"Processing failed: {str(e)}")
                continue
    
    logging.info(f"Successfully processed {successful} out of {len(video_urls)} videos")
