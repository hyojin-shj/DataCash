"""DataCash — 집에 있는 기기(폰·워치)로 촉각·시각·청각·몸 감각을 한 시간축에 모으는 로봇 학습 데이터 양산소.

모듈
  meta        Meta Model API (SAM 3.1 · Muse Spark · Muse Image · Muse Voice Transcribe) 클라이언트와 비용 장부
  session     녹화 세션 형식(폰·워치가 올리는 파일)과 합성 4감 세션 생성기
  sync        4감 자동 동기화 (시계 왕복 · 음향 비콘 · 진동 펄스 · 내용 상관 검증)
  touch       촉각 (워치 능동 진동으로 쥐는 힘, 영상으로 잡음·놓음, 접촉음)
  labels      손 21점 · 전신 33관절 (MediaPipe)
  privacy     SAM 3.1 가리기
  understand  Muse Spark 장면 이해와 감각 교차 검증
  export      LeRobot v3.0 4감 데이터셋 + 품질 인증서
  server      수집 서버 · 구매 기업용 웹 API
"""

__version__ = "0.1.0"
