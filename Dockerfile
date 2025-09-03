FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libopenblas-dev \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files
COPY requirements.txt ./

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
#COPY .env .env

RUN sed -i '1iimport sys\nsys.path.insert(0, "/app")\n' main.py

RUN find . -type d -exec sh -c 'touch "$0/__init__.py"' {} \;

ENV PYTHONPATH=/app

ENV DOC_FILE_PATH=/app/data/pdfs

CMD ["python3", "main.py"]
