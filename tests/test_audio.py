"""Audio sink buffer mechanics — CI-safe (no real device: we drive _callback directly)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from billy.systems.nes.retro_session import _AudioSink


def _live(rate=32000, channels=2):
    """A sink whose buffer path is enabled without opening a real PortAudio stream."""
    sink = _AudioSink(rate=rate, channels=channels)
    sink._stream = object()      # sentinel: feed() writes to the ring buffer, no device
    return sink


def test_feed_without_stream_is_a_noop():
    sink = _AudioSink(rate=32000)          # never started → _stream is None
    sink.feed(np.ones((534, 2), dtype=np.int16))
    assert len(sink._buf) == 0             # silently dropped, never raises


def test_callback_drains_buffer_then_pads_silence():
    sink = _live()
    sink.feed(np.arange(200, dtype=np.int16).reshape(100, 2))
    out = np.empty((60, 2), dtype=np.int16)
    sink._callback(out, 60, None, None)
    assert len(sink._buf) == 40            # 100 fed - 60 consumed
    assert out[0, 0] == 0                  # FIFO order preserved
    # underflow: ask for more than remain → the tail is zero-padded, no crash
    out2 = np.full((80, 2), 99, dtype=np.int16)
    sink._callback(out2, 80, None, None)
    assert len(sink._buf) == 0
    assert (out2[40:] == 0).all()          # padded region silenced


def test_overflow_drops_oldest_to_cap_latency():
    sink = _live(rate=32000)               # cap = 0.20s * 32000 = 6400 samples
    sink.feed(np.zeros((50000, 2), dtype=np.int16))
    assert len(sink._buf) <= sink._cap == 6400


def test_mono_is_upmixed_to_stereo():
    sink = _live()
    sink.feed(np.arange(10, dtype=np.int16))     # 1-D mono input
    assert sink._buf.shape == (10, 2)
    assert (sink._buf[:, 0] == sink._buf[:, 1]).all()
