# ===========================================
# PIILOT AI Server Dockerfile
# ===========================================
# Python 3.11 + 시스템 의존성 (ffmpeg, OpenCV 등)
# AI 모델은 이미지에 포함하지 않고, Docker Volume으로 캐시합니다.
# (처음 시작할 때 자동 다운로드, 이후엔 캐시에서 로드)
# ===========================================

FROM python:3.11-slim

# 시스템 의존성 설치
# - build-essential, libpq-dev: Python 패키지 빌드에 필요
# - libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender-dev: OpenCV (이미지 처리)
# - libgomp1: XGBoost (머신러닝)
# - ffmpeg: 오디오/비디오 처리 (Whisper STT, 비디오 마스킹)
# - curl: 헬스체크용
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (캐시 활용)
# requirements.txt가 바뀌지 않으면 pip install을 건너뜁니다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 출력 디렉토리 생성 (마스킹 결과 파일 저장용)
RUN mkdir -p /app/output_file/documents \
    /app/output_file/images \
    /app/output_file/videos \
    /app/output_file/audio

# 8000 포트 사용
EXPOSE 8000

# 헬스체크: 30초마다 /health 호출
# start_period: 120초 (AI 모델 로딩 시간이 길어서 2분 대기)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
