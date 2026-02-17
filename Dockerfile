# ===========================================
# PIILOT AI Server Dockerfile (GPU 버전)
# ===========================================
# NVIDIA CUDA 12.4 + cuDNN 기반 이미지
# g4dn.xlarge (T4 GPU) 에서 사용합니다.
# ===========================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 대화식 프롬프트 방지 (tzdata 등)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Python 3.11 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 버전 먼저 설치 (캐시 활용)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 출력 디렉토리 생성
RUN mkdir -p /app/output_file/documents \
    /app/output_file/images \
    /app/output_file/videos \
    /app/output_file/audio

EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FastAPI 서버 실행
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
