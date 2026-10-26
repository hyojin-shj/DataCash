"""녹화 세션 형식과 읽기.

폰(카메라 역할)과 워치가 각자 자기 시계(부팅 후 시계, ns)로 적은 파일을 한 폴더에 모은다. 시계를 맞추는 일은 sync 가 한다.

  meta.json            세션 id, 미션, 기기 기종, 동의, 녹화 중 '실수했어요'(폰 시계 ns), 미션 후 응답
  video.mp4            폰 카메라 영상
  video_frames.csv     frame, t_ns                (폰 시계, 프레임마다 촬영 시각)
  audio.wav            폰 마이크 (모노)
  audio.json           {"start_ns", "rate"}       (첫 샘플의 폰 시계 시각)
  phone_imu.csv        t_ns, ax, ay, az, gx, gy, gz           (폰 시계)
  beacon.csv           t_ns, freq0_hz, freq1_hz, dur_ms      폰 스피커가 낸 음향 비콘 (폰 시계, 낸다고 명령한 시각)
  watch_imu.csv        t_ns, ax, ay, az, gx, gy, gz           (워치 시계)
  watch_audio.wav      워치 마이크 (선택)
  watch_audio.json     {"start_ns", "rate"}       (워치 시계)
  watch_vib.csv        t_ns, dur_ms, amp          워치 능동 진동 펄스 (워치 시계, 울리라고 명령한 시각)
  watch_hr.csv         t_ns, bpm                  (워치 시계, 생체 동의 시)
  watch_skin.csv       t_ns, temp_c               (워치 시계, 생체 동의 시)
  clock_sync.csv       phone_send_ns, watch_ns, phone_recv_ns   폰↔워치 시각 주고받기 (왕복)
"""

from .format import Session, load_session

__all__ = ["Session", "load_session"]
