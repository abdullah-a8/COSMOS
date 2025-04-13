import fitz
from newspaper import Article
from time import sleep
import hashlib
from youtube_transcript_api import YouTubeTranscriptApi

def extract_text_from_pdf(file):
    try:
        file_content = file.read()
        pdf_hash = hashlib.sha256(file_content).hexdigest()
        pdf_document = fitz.open("pdf", file_content)
        all_text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            all_text += page.get_text()
        pdf_document.close()
        return all_text, pdf_hash
    except Exception as e:
        return f"Error reading PDF: {e}", None

def extract_text_from_url(url, retries=3):
    for attempt in range(retries):
        try:
            article = Article(url)
            article.download()
            article.parse()

            if len(article.text.strip()) == 0:
                raise ValueError("No text extracted. The article might be behind a paywall or inaccessible.")

            return article.text, url
        except Exception as e:
            if attempt < retries - 1:
                sleep(2)
                continue
            return f"Error processing URL after {retries} attempts: {e}", None

def extract_transcript_details(youtube_video_url):
    try:
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_video_url:
            video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return "Error: Invalid YouTube URL format.", None

        print(f"Extracting transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = " ".join([i["text"] for i in transcript_list])

        if not transcript:
            return "Error: Could not retrieve transcript (may be disabled for this video).", f"youtube_{video_id}"

        return transcript, f"youtube_{video_id}"

    except Exception as e:
        print(f"Error in YouTube transcript extraction: {e}")
        video_id_on_error = None
        if 'video_id' in locals():
            video_id_on_error = f"youtube_{video_id}"
        return f"Error retrieving transcript: {e}", video_id_on_error 