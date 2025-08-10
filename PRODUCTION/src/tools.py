tools = [
    {
        "type": "function",
        "function": {
            "name": "cleanQuery",
            "description": "Takes in a query, and returns a json object with the following keys: 'websites', 'youtube', and 'cleaned_query'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The initial raw query mentioned made by the user"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a search using either Google, DuckDuckGo, or Mojeek based on the query and returns up to 5 result URLs as a list",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_metadata",
            "description": "Retrieves the title and duration of a YouTube video from its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the YouTube video."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_transcript",
            "description": "Fetches the transcript of a YouTube video, truncated if too long.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the YouTube video."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_full_text",
            "description": "Fetches and parses a web page, extracting text, title, and up to three main images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the web page to fetch."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "Returns the current local time and UTC time and location for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {"type": "string", "description": "The name of the location to get the time for."}
                },
                "required": ["location_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_prompt_from_image",
            "description": "Generates a search-friendly prompt from an image in base64 format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imageURL": {"type": "string", "description": "The URL of the image"}
                },
                "required": ["imageURL"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replyFromImage",
            "description": "Generates a friendly response based on an image and a user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "imageURL": {"type": "string", "description": "The URL of the image"},
                    "query": {"type": "string", "description": "The user's query related to the image."}
                },
                "required": ["imageURL", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Performs an image search using Google and returns up to 10 image URLs and their sources as lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_query": {"type": "string", "description": "The search query string for images."},
                    "max_images" : {"type": "integer", "description": "The maximum number of images to return."}
                },
                "required": ["image_query"]
            }
        }
    }
]
