# Hugging Face Space (Docker) — DataCash 서버 + 구매 기업 웹. GPU 없음: 얼굴 치환은 Baseten (DATACASH_FACESWAP=baseten, 실패하면 흐리게).
# 묶음은 scripts/build_space.py 가 만든다 (코드 + 공개해도 되는 시연 데이터만 seed/ 에).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --no-deps -e .

USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 DATACASH_DATA=/home/user/data DATACASH_FACESWAP=baseten
EXPOSE 7860
CMD ["bash", "start.sh"]
