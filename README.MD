# Scribly

Below is a sample README in Markdown format that you can use for your project:

---

# YouTube Video Content Repurposer

A command-line tool to transcribe YouTube videos and automatically repurpose the content into summaries, blog posts, and social media snippets. Perfect for content creators looking to quickly transform their video content for various platforms!

## Features

- **Video Transcription:**  
  Uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe audio from YouTube videos.
  
- **Audio Standardization:**  
  Converts downloaded audio into a standardized WAV format (mono, 16 kHz, 32-bit float) for consistent transcription quality.
  
- **Content Repurposing:**  
  Leverages [Hugging Face Transformers](https://huggingface.co/transformers/) to generate:
  - A concise summary of the transcript.
  - A detailed blog post based on the transcript.
  - A catchy social media snippet.

- **Playlist Support & Parallel Processing:**  
  Supports processing entire YouTube playlists and uses a thread pool for faster transcription.

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:JUSTSUJAY/scribly.git
   cd scribly
   ```

2. **Create a virtual environment and activate it (optional but recommended):**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure `ffmpeg` is installed and available in your PATH.**  
   (If not, adjust the `get_ffmpeg_path()` function in `transcriber.py` with your system's ffmpeg executable path.)

    *to add it in your env variable run*
    ```
    export FFMPEG_PATH=/path/to/
    ```
## Usage

The project is run from the command line. The main entry point is `src/main.py`.

### Transcribe Videos Only

To transcribe a YouTube video or playlist:

```bash
python src/main.py "https://youtube.com/playlist?list=YOUR_PLAYLIST_ID" --model base
```

### Transcribe & Repurpose Content

To transcribe and generate a summary, blog post, and social media snippet from each video:

```bash
python src/main.py "https://youtube.com/playlist?list=YOUR_PLAYLIST_ID" --model base --repurpose
```

### Command-Line Arguments

- **links:** One or more YouTube video or playlist URLs.
- **--model:** Specify the Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large` (default is `base`).
- **--workers:** Number of parallel threads for processing videos (default: 4).
- **--output:** Directory where transcripts and repurposed content will be saved (default: `transcripts`).
- **--repurpose:** Flag to generate summary, blog post, and social snippet from each transcript.

## Project Structure

```
scribly/
├── src/
│   ├── main.py             
│   ├── transcriber.py       
│   ├── repurposer.py        
│   └── utils.py            
├── requirements.txt        
└── README.md               
```

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. When contributing, please follow the existing code style and include tests when possible.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for robust speech recognition.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for state-of-the-art text generation.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading audio from YouTube.
- Thanks to the community for all the support and contributions!

---
