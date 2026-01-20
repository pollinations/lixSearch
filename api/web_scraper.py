"""
Web Scraper Module - Re-exports fetch_full_text from search module
This maintains backward compatibility with the existing import structure
"""

from search import fetch_full_text

__all__ = ['fetch_full_text']


