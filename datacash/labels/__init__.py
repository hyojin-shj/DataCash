"""손 21점 × 양손, 전신 33관절(화면 좌표 + 3D 미터) — MediaPipe (Apache 2.0). 가리기 전 원본(폰 안 또는 동의자 영상)에서 뽑는다."""

from .pose import FrameLabels, extract_labels

__all__ = ["FrameLabels", "extract_labels"]
