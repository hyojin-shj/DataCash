import json

import numpy as np
import pytest

from datacash.session.synth import generate


@pytest.fixture(scope="session")
def synth_session(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth") / "s1"
    truth = generate(d)
    return d, truth


@pytest.fixture(scope="session")
def synth_tracks(synth_session):
    d, _ = synth_session
    tr = json.loads((d / "tracks.json").read_text())
    hand = [np.array([[h[0] + 16 * np.cos(a), h[1] + 16 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)])
            for h in [x["hand"] for x in tr]]
    objs = [{0: tuple(x["cup"])} for x in tr]
    return hand, objs, [x["t_s"] for x in tr]


def pytest_sessionstart(session):
    """Numba 해독기를 별도 프로세스에서 미리 컴파일 (같은 프로세스의 OpenCV·MediaPipe 스레드와 엉켜 멈추지 않게)."""
    from datacash.meta.mask_fast import warm

    warm()
