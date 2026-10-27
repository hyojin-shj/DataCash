"""만든 데이터셋을 실제 LeRobot 으로 읽어 본다 (별도 환경: data/lrvenv, lerobot[dataset] 0.6.1).

  data/lrvenv/bin/python scripts/validate_lerobot.py out/synth_s1/lerobot
영상은 pyav 로 푼다 (torchcodec 은 시스템 FFmpeg 라이브러리가 필요).
"""

import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(root):
    ds = LeRobotDataset("local/datacash", root=root, video_backend="pyav")
    print(f"에피소드 {ds.num_episodes} · 장면 {len(ds)} · {ds.fps}fps")
    x = ds[len(ds) // 2]
    for k, v in x.items():
        print(f"  {k:32s} {tuple(v.shape) if hasattr(v, 'shape') else v}")


if __name__ == "__main__":
    main(sys.argv[1])
