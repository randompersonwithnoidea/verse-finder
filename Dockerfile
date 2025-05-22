FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt update && apt install -y python3 python3-pip ffmpeg

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install torch + other dependencies
RUN pip install --no-cache-dir \
    torch==2.7.0+cu118 \
    torchvision==0.22.0+cu118 \
    torchaudio==2.7.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy your code
WORKDIR /app
COPY . .

# Install rest of your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your app
CMD ["python", "main.py"]
