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
        ffmpeg_path = os.getenv('FFMPEG_PATH')  # Check environment variable first
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
        # If the file wasn't saved as .m4a, try converting from webm if available
        if not audio_filepath.exists():
            webm_file = output_path / f"{video_id}.webm"
            if webm_file.exists():
                ffmpeg_command = [
                    get_ffmpeg_path(), "-i", str(webm_file),
                    "-c:a", "aac", str(audio_filepath)
                ]
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                webm_file.unlink()  # Remove webm file after conversion
        return audio_filepath if audio_filepath.exists() else None
    except subprocess.CalledProcessError:
        logging.error(f"Failed to download audio for {video_url}")
        return None

def convert_audio_to_wav(audio_file: Path) -> Path:
    """
    Convert the downloaded audio file to a standardized WAV format:
      - Mono audio (1 channel)
      - 16 kHz sample rate
      - 32-bit float PCM (matches what Whisper expects)
    """
    output_audio = audio_file.with_suffix('.wav')
    ffmpeg_path = get_ffmpeg_path()
    command = [
        ffmpeg_path, '-y', '-i', str(audio_file),
        '-ac', '1', '-ar', '16000',
        '-acodec', 'pcm_f32le',  # 32-bit float PCM
        str(output_audio)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if output_audio.exists():
            logging.info(f"Converted audio to standardized format: {output_audio}")
            return output_audio
        else:
            logging.error(f"Conversion failed; output file not found: {output_audio}")
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert audio: {e}")
        return None

def transcribe_audio(audio_file: Path, model: whisper.Whisper) -> str:
    """Transcribes audio using Whisper's low-level functions."""
    if not audio_file or not audio_file.exists():
        return ""
    try:
        logging.info(f"Transcribing {audio_file}")
        # Load the standardized WAV file (which is now float32)
        audio = whisper.load_audio(str(audio_file))
        # Pad/trim to 30 seconds (Whisper’s expected input length)
        audio = whisper.pad_or_trim(audio)
        # Compute the log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # Decode (here we disable fp16 since we're likely on CPU)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        return ""

def process_video(video_url: str, output_dir: Path, model: whisper.Whisper) -> bool:
    """Process a single video: download, convert, and transcribe."""
    video_id = video_url.split("v=")[-1]
    video_dir = output_dir / video_id
    video_dir.mkdir(exist_ok=True, parents=True)
    
    audio_filepath = download_video_audio(video_url, video_dir)
    if audio_filepath and audio_filepath.exists():
        # Convert the audio file to a standardized WAV file (float32, mono, 16kHz)
        standardized_audio = convert_audio_to_wav(audio_filepath)
        if standardized_audio and standardized_audio.exists():
            transcript_text = transcribe_audio(standardized_audio, model)
            if transcript_text:
                transcript_file = video_dir / f"{video_id}_transcript.txt"
                transcript_file.write_text(transcript_text, encoding="utf-8")
                logging.info(f"Successfully transcribed: {video_id}")
                return True
    logging.error(f"Failed to process video: {video_id}")
    return False

def bulk_transcribe_videos(video_urls: list, model_name: str = "base", 
                           workers: int = 4, output_dir: str = "transcripts"):
    """Bulk transcribe videos with progress tracking."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

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
