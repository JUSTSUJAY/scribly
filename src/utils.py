import subprocess
import logging

def expand_playlist_urls(urls: list) -> list:
    """Expands playlist URLs into individual video URLs."""
    expanded_urls = []
    
    for url in urls:
        if "playlist" in url:
            command = [
                "yt-dlp",
                "--flat-playlist",
                "--get-id",
                "--ignore-errors",
                url
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                video_ids = [vid for vid in result.stdout.strip().split('\n') if vid]
                expanded_urls.extend([f"https://youtube.com/watch?v={vid}" for vid in video_ids])
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to expand playlist {url}: {e}")
        else:
            expanded_urls.append(url)
            
    return expanded_urls
