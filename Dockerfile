FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY assets ./assets
COPY data ./data
ENV PORT=8050
EXPOSE 8050
CMD gunicorn app:server --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}
