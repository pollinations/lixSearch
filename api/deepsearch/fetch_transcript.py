import yt_dlp
import os
import requests
import json


# def download_audio(url, output_folder="downloads"):
#     os.makedirs(output_folder, exist_ok=True)

#     ydl_opts = {
#         "format": "bestaudio/best",
#         "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
#         "cookiefile": None,  # not needed when using cookies-from-browser
#         "cookiesfrombrowser": ("chrome",),  # or ("brave",), ("edge",), ("firefox",)
#         "postprocessors": [
#             {
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "mp3",
#                 "preferredquality": "192",
#             }
#         ],
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         filename = ydl.prepare_filename(info)
#         mp3_file = os.path.splitext(filename)[0] + ".mp3"
#         return mp3_file



# if __name__ == "__main__":
#     url = "https://www.youtube.com/api/timedtext?v=_160oMzblY8&ei=GrMladHdCcTS3LUPiMDgsQ4&caps=asr&opi=112496729&exp=xpe&xoaf=5&xowf=1&hl=en-GB&ip=0.0.0.0&ipbits=0&expire=1764103562&sparams=ip%2Cipbits%2Cexpire%2Cv%2Cei%2Ccaps%2Copi%2Cexp%2Cxoaf&signature=96A7BDB685CFAC3FA524EB4622AE1D60A9167754.5FB577C86E4BAA7C5221AB6BD15CBF89BDCC4783&key=yt8&lang=en&potc=1&pot=MlNdhAg5hq9gT124ZeDzzRH4OGstiX0F3zZ_vOWjl888E15192kw8w4DWWW5hgclLltBMJ3B1QBTx1VzA7Mc_lHleG6Ax6wF0ggKfh3I-1mpkO0pqA%3D%3D&fmt=json3&xorb=2&xobt=3&xovt=3&cbr=Chrome&cbrver=142.0.0.0&c=WEB&cver=2.20251124.01.00&cplayer=UNIPLAYER&cos=Windows&cosver=10.0&cplatform=DESKTOP"

#     response = requests.get(url)
#     data = response.json()

#     def extract_transcript_segments(data):
#         segments = []
#         def recurse(obj):
#             if isinstance(obj, dict):
#                 if "segs" in obj:
#                     for seg in obj["segs"]:
#                         if "utf8" in seg:
#                             segments.append(seg["utf8"].strip())
#                 for v in obj.values():
#                     recurse(v)
#             elif isinstance(obj, list):
#                 for item in obj:
#                     recurse(item)
#         recurse(data)
#         return " ".join(segments)

#     transcript_text = extract_transcript_segments(data)
#     print(transcript_text)