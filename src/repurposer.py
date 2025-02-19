import logging
from pathlib import Path
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def generate_summary(text: str, max_length=150, min_length=40) -> str:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Enable truncation so that the input doesn't exceed model limits.
    summary = summarizer(
        text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False, 
        truncation=True
    )
    return summary[0]['summary_text']

def generate_blog_post(text: str, max_new_tokens=200) -> str:
    generator = pipeline("text-generation", model="gpt2")
    prompt = "Write a detailed blog post based on the following transcript:\n" + text
    blog_post = generator(
        prompt, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=0.7, 
        truncation=True
    )
    return blog_post[0]['generated_text']

def generate_social_snippet(text: str, max_new_tokens=50) -> str:
    generator = pipeline("text-generation", model="gpt2")
    prompt = "Generate a catchy social media snippet based on the following transcript:\n" + text
    snippet = generator(
        prompt, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=0.8, 
        truncation=True
    )
    return snippet[0]['generated_text']

def repurpose_transcript(transcript_file: Path):
    if not transcript_file.exists():
        logging.error(f"Transcript file not found: {transcript_file}")
        return
    text = transcript_file.read_text(encoding="utf-8")
    
    logging.info("Generating summary...")
    summary = generate_summary(text)
    
    logging.info("Generating blog post...")
    blog_post = generate_blog_post(text)
    
    logging.info("Generating social snippet...")
    snippet = generate_social_snippet(text)

    # Save repurposed content alongside the transcript
    folder = transcript_file.parent
    (folder / "summary.txt").write_text(summary, encoding="utf-8")
    (folder / "blog_post.txt").write_text(blog_post, encoding="utf-8")
    (folder / "social_snippet.txt").write_text(snippet, encoding="utf-8")
    
    logging.info(f"Repurposed content saved in {folder}")

def repurpose_all(transcripts_dir: str = "transcripts"):
    base_path = Path(transcripts_dir)
 
