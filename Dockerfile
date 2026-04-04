FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)); raise SystemExit(0 if data.get('status') in {'ok', 'degraded'} else 1)"

ENTRYPOINT ["python", "flask_api_multi.py"]
CMD ["--host", "0.0.0.0", "--port", "5000", "--mode", "multi"]
