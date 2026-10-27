"""맥 GPU(MPS)에서 LivePortrait 를 빠르게 — 큰 3D 합성곱을 2D 합성곱 여러 번으로.

LivePortrait 움직임 변형(dense_motion)의 mask 층은 7×7×7 Conv3d 인데, MPS 의 Conv3d 가 이 크기에서 아주 느려
장면당 1.3초를 썼다 (얼굴 치환 시간의 80%). 깊이 방향 7칸을 하나씩 떼어 2D 합성곱(배치 = 깊이)으로 더하면
같은 값(차이 1e-5)에 16배 빠르다. 다른 장치·작은 커널은 원래대로."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def conv3d_as_2d(x: torch.Tensor, weight: torch.Tensor, bias, padding: int) -> torch.Tensor:
    n, c, d, h, w = x.shape
    k = weight.shape[2]
    xs = F.pad(x, (0, 0, 0, 0, padding, padding)).permute(0, 2, 1, 3, 4)       # (n, d+2p, c, h, w)
    out = None
    for dz in range(k):
        y = F.conv2d(xs[:, dz:dz + d].reshape(n * d, c, h, w), weight[:, :, dz], padding=padding)
        out = y if out is None else out + y
    out = out.reshape(n, d, -1, h, w).permute(0, 2, 1, 3, 4)
    return out if bias is None else out + bias.view(1, -1, 1, 1, 1)


def _forward(self, x):
    if x.device.type == "mps":
        return conv3d_as_2d(x, self.weight, self.bias, self.padding[0])
    return torch.nn.Conv3d.forward(self, x)


def speed_up(model: torch.nn.Module, min_kernel: int = 5) -> int:
    """model 안의 큰 정육면체 Conv3d (stride 1, 같은 padding) 를 바꿔 낀다. 바꾼 층 수."""
    n = 0
    for m in model.modules():
        if (isinstance(m, torch.nn.Conv3d) and m.kernel_size[0] >= min_kernel and len(set(m.kernel_size)) == 1
                and m.stride == (1, 1, 1) and m.dilation == (1, 1, 1) and m.groups == 1
                and len(set(m.padding)) == 1 and m.padding[0] == m.kernel_size[0] // 2 and m.padding_mode == "zeros"):
            m.forward = types.MethodType(_forward, m)
            n += 1
    return n
