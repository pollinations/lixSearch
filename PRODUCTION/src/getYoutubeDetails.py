from urllib.parse import urlparse, parse_qs
import re
from conditional_print import conditional_print
from pytube import YouTube, exceptions
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter

get_youtube_video_metadata_show_log = False
MAX_TRANSCRIPT_WORD_COUNT = 5000


def get_youtube_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
        if parsed_url.path:
            match = re.search(r'/(?:embed|v)/([^/?#&]+)', parsed_url.path)
            if match:
                return match.group(1)
    elif "youtu.be" in parsed_url.netloc:
        path = parsed_url.path.lstrip('/')
        if path:
            video_id = path.split('/')[0].split('?')[0].split('#')[0]
            video_id = video_id.split('&')[0]
            return video_id
    return None


def get_youtube_video_metadata(url, show_logs=get_youtube_video_metadata_show_log):
    video_id = get_youtube_video_id(url)
    if not video_id:
        conditional_print(f"[Pytube] Invalid URL provided for metadata: {url}", show_logs)
        return None

    try:
        yt = YouTube(url)
        metadata = {
            "title": yt.title if hasattr(yt, 'title') else "Unknown",
            "author": yt.author if hasattr(yt, 'author') else "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S") if hasattr(yt, 'publish_date') and yt.publish_date else "Unknown",
            "length": f"{yt.length // 60}m {yt.length % 60}s" if hasattr(yt, 'length') and yt.length is not None else "Unknown",
            "views": f"{yt.views:,}" if hasattr(yt, 'views') and yt.views is not None else "Unknown",
            "description": (
                yt.description[:500] + "..." if hasattr(yt, 'description') and yt.description and len(yt.description) > 500
                else getattr(yt, 'description', "No description available") or "No description available"
            ),
            "thumbnail_url": yt.thumbnail_url if hasattr(yt, 'thumbnail_url') else None,
            "url": url
        }
        conditional_print(f"[Pytube] Fetched metadata for {url}", show_logs)
        return metadata

    except exceptions.VideoUnavailable:
        conditional_print(f"[Pytube] VideoUnavailable: {url}", show_logs)
    except exceptions.LiveStreamError:
        conditional_print(f"[Pytube] LiveStreamError (likely a live stream or not fully processed yet): {url}", show_logs)
    except exceptions.RegexMatchError:
        conditional_print(f"[Pytube] RegexMatchError: Invalid or malformed URL - {url}", show_logs)
    except exceptions.ExtractError:
        conditional_print(f"[Pytube] ExtractError: Could not extract data for - {url}", show_logs)
    except Exception as e:
        conditional_print(f"[Pytube] Unexpected error for {url}: {type(e).__name__} - {e}", show_logs)

    return None


def get_youtube_transcript(video_id, show_logs=True):
    if not video_id:
        conditional_print("Attempted to get transcript with no video ID.", show_logs)
        return None

    try:
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            conditional_print(f"Found English ('en') transcript for video ID: {video_id}", show_logs)
        except NoTranscriptFound:
            conditional_print(f"No 'en' transcript found. Trying other available languages.", show_logs)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available = list(transcript_list._manually_created_transcripts.values()) + list(transcript_list._generated_transcripts.values())
            if not available:
                conditional_print(f"No transcripts found in any language for video ID: {video_id}", show_logs)
                return None
            transcript = available[0]
            conditional_print(f"Using transcript in '{transcript.language_code}'", show_logs)
            entries = transcript.fetch()

        if not entries:
            raise ValueError("Transcript fetch returned no entries.")
        full_text = " ".join(entry['text'] for entry in entries)
        
        words = full_text.split()
        if len(words) > MAX_TRANSCRIPT_WORD_COUNT:
            conditional_print(f"Transcript length ({len(words)} words) exceeds MAX_TRANSCRIPT_WORD_COUNT ({MAX_TRANSCRIPT_WORD_COUNT}). Truncating.", show_logs)
            return " ".join(words[:MAX_TRANSCRIPT_WORD_COUNT]) + "..."
        return full_text

    except NoTranscriptFound:
        conditional_print(f"No transcript available for video ID: {video_id}", show_logs)
    except TranscriptsDisabled:
        conditional_print(f"Transcripts are disabled for video ID: {video_id}", show_logs)
    except Exception as e:
        conditional_print(f"Unexpected error while fetching transcript for {video_id}: {type(e).__name__} - {e}", show_logs)

    return None

