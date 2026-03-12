FROM python:3.12-slim

# Install useful OS packages for development/debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/src

# Install Python dependencies first for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    if [ -f /tmp/requirements.txt ]; then pip install -r /tmp/requirements.txt; fi

# Keep container alive for VS Code attach
CMD ["sleep", "infinity"]