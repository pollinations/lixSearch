lsof -ti:5000 | xargs kill -9 && \
docker system prune -a -f && \
docker compose down && \
docker compose build --no-cache && \
docker compose up -d --build 