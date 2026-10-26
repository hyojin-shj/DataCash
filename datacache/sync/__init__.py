"""4감 자동 동기화 — 사용자 동작(손목 톡톡) 없이 폰 시계 하나로 모든 감각을 맞춘다.

  ① 같은 기기 안: 폰 카메라·마이크·IMU 는 원래 같은 시계(BOOTTIME)
  ② clock    폰↔워치 시각 주고받기, 왕복이 짧은 것만 써서 직선(오프셋 + 드리프트)
  ③ latency  기기 모델별 고정 지연표 (사람이 아니라 기종마다 한 번 잰 값)
  ④ beacon   폰 스피커의 18~20kHz 처프를 폰·워치 마이크가 같이 듣고 맞춘다 (BeepBeep 의 자기 녹음 원리: 스피커 지연이 빠진다)
     vibpulse 워치 능동 진동(촉각용) 펄스를 워치 IMU 와 폰 마이크가 같이 잡는다 — 촉각 펄스가 동기 신호를 겸한다
  ⑤ content  영상 속 움직임 ↔ 손목 IMU, 폰 소리 ↔ 워치 소리 상관으로 검증 (어긋나면 그 채널을 판매에서 뺀다)
"""

from .align import Alignment, align_session

__all__ = ["Alignment", "align_session"]
