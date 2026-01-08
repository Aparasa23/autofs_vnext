FROM python:3.10-slim

WORKDIR /app

# System deps for common scientific wheels (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Install as package (so entrypoint works)
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["autofs-vnext"]
CMD ["list-methods"]
