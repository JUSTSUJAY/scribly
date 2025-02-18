import argparse
from transcriber import bulk_transcribe_videos
from utils import expand_playlist_urls

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bulk transcribe YouTube videos from a playlist.")
    parser.add_argument("links", nargs="+", help="YouTube video or playlist links.")
    parser.add_argument("--model", default="base", choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help="Specify the Whisper model size")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads for transcription.")
    parser.add_argument("--output", default="transcripts", help="Output directory for transcripts")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Expand playlist URLs into individual video URLs
    video_urls = expand_playlist_urls(args.links)
    bulk_transcribe_videos(video_urls, model_name=args.model, 
                          workers=args.workers, output_dir=args.output)

if __name__ == "__main__":
    main()
