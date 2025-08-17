FROM python:3.12-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install playwright 
RUN playwright install chromium 
RUN playwright install-deps  
RUN pip install ddgs

COPY . .

EXPOSE 5000
ENV FLASK_APP=src/app.py
CMD ["python3", "src/app.py"]
