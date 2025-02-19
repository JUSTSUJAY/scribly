import argparse
from transcriber import bulk_transcribe_videos
from utils import expand_playlist_urls
from repurposer import repurpose_all

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bulk transcribe and repurpose YouTube videos from a playlist."
    )
    parser.add_argument("links", nargs="+", help="YouTube video or playlist links.")
    parser.add_argument("--model", default="base", choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help="Specify the Whisper model size")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads for transcription.")
    parser.add_argument("--output", default="transcripts", help="Output directory for transcripts")
    parser.add_argument("--repurpose", action="store_true", 
                        help="Generate summary, blog post, and social snippet from transcripts")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Expand playlist URLs (if applicable) to individual video URLs
    video_urls = expand_playlist_urls(args.links)
    
    # Transcribe videos
    bulk_transcribe_videos(video_urls, model_name=args.model, 
                           workers=args.workers, output_dir=args.output)
    
    # Optionally, repurpose the transcripts (summary, blog post, social snippet)
    if args.repurpose:
        repurpose_all(args.output)

if __name__ == "__main__":
    main()
