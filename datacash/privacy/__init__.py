"""개인정보 가리기 — SAM 3.1 글자 지시 분할로 '동의 안 한 것'만 정확히 가린다.

  사람·아이     참여자(동의자) 한 명만 남기고 나머지 사람 영역 전체를 회색으로 (얼굴만 블러하면 몸이 그대로 남는다)
  화면·거울·액자·서류   글자 지시로 분할해 회색 (화면 속 사람·사진 속 얼굴·글자)
  조작 물체      가리지 않고 영역만 기록 → LeRobot observation.objects
참여자 얼굴은 여기서 가리지 않는다 — 얼굴 치환(실사 가상 인물) 단계가 따로 맡는다.
"""

from .mask import PrivacyResult, mask_video

__all__ = ["PrivacyResult", "mask_video"]
