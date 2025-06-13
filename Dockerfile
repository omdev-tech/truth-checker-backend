# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN pip install yt-dlp

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy cookies file (if exists)
COPY youtube_cookies.txt ./youtube_cookies.txt

# Set environment variables for production
ENV PYTHONPATH=/app
ENV YT_DLP_ENABLE_COOKIES=true
ENV YT_DLP_COOKIES_FILE=youtube_cookies.txt

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "truth_checker.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 