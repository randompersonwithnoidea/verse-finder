FROM python:3.13-slim

# Install system dependencies
RUN apt update && apt install -y ffmpeg

# Set working directory
WORKDIR /app

# Copy code
COPY . .

# Install dependencies (CPU-only torch)
RUN pip install --no-cache-dir \
    torch==2.7.0+cpu \
    torchvision==0.22.0+cpu \
    torchaudio==2.7.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install the rest from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set command
CMD ["python", "main.py"]
