import re
import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    return logging.getLogger(name)


def validate_query(query: str, max_length: int = 5000) -> bool:
    """Validate user query."""
    if not query or not isinstance(query, str):
        return False
    if len(query) > max_length:
        return False
    if len(query.strip()) == 0:
        return False
    return True


def validate_session_id(session_id: str, pattern: str = r'^[a-zA-Z0-9\-]{8,36}$') -> bool:
    """Validate session ID format."""
    if not session_id or not isinstance(session_id, str):
        return False
    return bool(re.match(pattern, session_id))


def validate_url(url: str, max_length: int = 2048) -> bool:
    """Validate URL format."""
    if not url or not isinstance(url, str):
        return False
    if len(url) > max_length:
        return False
    if not url.startswith(('http://', 'https://')):
        return False
    return True
