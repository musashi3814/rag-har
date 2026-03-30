# RTX 4090 対応（CUDA 11.8 + Python 3.10）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1

# --- 1. Ubuntuリポジトリ修正 ---

# --- 2. 依存パッケージ ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    pkg-config \
    libhdf5-dev \
    ffmpeg \
    unzip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 3. Python設定 ---
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# --- 5. 依存ライブラリの事前インストール ---
RUN cat /app/requirements.txt
RUN pip install --no-cache-dir numpy 

# --- 6. requirements.txt のインストール ---
# RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# --- 10. アプリケーション配置 ---
COPY . /app
COPY data /app/data

# --- 11. PYTHONPATH設定 ---
ENV PYTHONPATH="/app:${PYTHONPATH}"

# --- 12. Jupyter設定 ---
EXPOSE 8888
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
