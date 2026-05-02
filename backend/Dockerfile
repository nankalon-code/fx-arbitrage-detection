FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
