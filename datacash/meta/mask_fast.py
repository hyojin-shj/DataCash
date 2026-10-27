"""SAM 3.1 마스크(one_bit) 빠른 해독 — 공식 meta_sam_parser 와 같은 산술 부호 해독을 Numba 로.

공식 파서는 픽셀마다 파이썬으로 돌고, 검사하려고 '풀기 + 다시 묶기'를 하며, 응답을 읽을 때 한 번·우리가 쓸 때 한 번 또 한다
→ 마스크 하나에 약 3초, 13초 녹화의 가리기 단계가 10분 (점심 녹화). 여기서는 한 번만, 기계어로 푼다.
결과는 공식 해독기와 비트 단위로 같다 (tests/test_sam_thin.py). lossless 는 공식 해독기로."""

from __future__ import annotations

import numpy as np
from numba import njit

_TOP, _BOTTOM, _M32 = 1 << 24, 1 << 16, 0xFFFFFFFF
_INCREMENT, _COUNT_LIMIT, _CONTEXT_COUNT, _PADDING = 14, 4096, 1 << 12, 2


@njit(cache=True)
def _decode(inp, width, height):
    counts = np.ones(_CONTEXT_COUNT * 2, np.int64)
    pw = width + 2 * _PADDING
    padded = np.zeros(pw * (height + _PADDING), np.uint8)
    out = np.zeros(width * height, np.uint8)
    n = inp.shape[0]
    if n < 4:
        return out, False
    pos, low, rng, code = 0, 0, _M32, 0
    for _ in range(4):
        code = ((code << 8) | inp[pos]) & _M32
        pos += 1
    for y in range(height):
        row = (y + _PADDING) * pw + _PADDING
        for x in range(width):
            p = row + x
            pr = p - pw
            ea = p - 2 * pw
            ctx = (padded[p - 1] | (padded[p - 2] << 1) | (padded[pr - 2] << 2) | (padded[pr - 1] << 3)
                   | (padded[pr] << 4) | (padded[pr + 1] << 5) | (padded[pr + 2] << 6) | (padded[ea - 2] << 7)
                   | (padded[ea - 1] << 8) | (padded[ea] << 9) | (padded[ea + 1] << 10) | (padded[ea + 2] << 11))
            off = ctx << 1
            z = counts[off]
            o = counts[off + 1]
            tot = z + o
            scaled = rng // tot
            if scaled == 0:
                return out, False
            value = ((code - low) & _M32) // scaled
            if value >= tot:
                value = tot - 1
            if value < z:
                bit, cum, freq = 0, 0, z
            else:
                bit, cum, freq = 1, z, o
            low = (low + scaled * cum) & _M32
            rng = scaled * freq
            while True:
                if ((low ^ ((low + rng) & _M32)) & _M32) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _M32) & (_BOTTOM - 1)
                else:
                    break
                if pos >= n:
                    return out, False
                code = ((code << 8) | inp[pos]) & _M32
                pos += 1
                low = (low << 8) & _M32
                rng = (rng << 8) & _M32
            padded[p] = bit
            out[y * width + x] = bit
            counts[off + bit] += _INCREMENT
            if counts[off] + counts[off + 1] >= _COUNT_LIMIT:
                counts[off] = max(counts[off] >> 1, 1)
                counts[off + 1] = max(counts[off + 1] >> 1, 1)
    return out, True


def warm() -> None:
    """컴파일을 별도 프로세스에서 미리 해 캐시(__pycache__)에 남긴다.
    OpenCV·MediaPipe 스레드가 떠 있는 프로세스에서 Numba 가 처음 컴파일하면 서로 기다리다 멈춘 적이 있다 (서버·테스트 시작 때 부른다)."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", "import numpy as np; from datacash.meta.mask_fast import _decode; "
                    "_decode(np.zeros(8, np.uint8), 1, 1)"], timeout=300, check=False)


def decode(mask) -> np.ndarray:
    """SegmentationMask → (height, width) uint8 0/1."""
    from meta_sam_parser._mask_codec import _unpack, decode_mask_to_raster

    if mask.encoding != "one_bit":
        return np.frombuffer(decode_mask_to_raster(mask), np.uint8).reshape(mask.height, mask.width)
    if not mask.payload.startswith("!"):
        raise ValueError("one_bit mask payloads must start with !")
    raster, ok = _decode(np.frombuffer(_unpack(mask.payload[1:]), np.uint8), int(mask.width), int(mask.height))
    if not ok:
        raise ValueError("SAM 마스크를 풀지 못함")
    return raster.reshape(mask.height, mask.width)
