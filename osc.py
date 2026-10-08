"""
osc.py - send tracked positions as OSC messages over UDP (no extra packages).

Each frame, one message per tracked person or object:

  /person  id x y     id: int, stable while the person is tracked
                      x, y: float, metres on the floor from the first tape mark
                      (0..width along the tape's first side, 0..height along the other)

Nothing is sent when someone leaves: a receiver should drop anyone it hasn't
heard about for a moment (the Processing sketches use half a second). That also
copes with lost UDP packets.
"""

import socket
import struct


def add_osc_args(p, default_port=12000):
    p.add_argument("--osc-host", default="127.0.0.1",
                   help="where to send OSC (default: 127.0.0.1, this computer)")
    p.add_argument("--osc-port", type=int, default=default_port,
                   help=f"OSC port (default: {default_port}, what the Processing sketches listen on)")


def _padded(b):
    """OSC strings end with at least one zero byte, padded to a multiple of 4."""
    return b + b"\0" * (4 - len(b) % 4)


def osc_message(address, *args):
    """Encode one OSC message. Arguments may be int, float or str."""
    tags, data = ",", b""
    for a in args:
        if isinstance(a, bool):
            raise TypeError("OSC bools aren't supported here; send an int")
        if isinstance(a, int):
            tags += "i"
            data += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            data += struct.pack(">f", a)
        elif isinstance(a, str):
            tags += "s"
            data += _padded(a.encode())
        else:
            raise TypeError(f"can't send {type(a).__name__} over OSC")
    return _padded(address.encode()) + _padded(tags.encode()) + data


class OscSender:
    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, address, *args):
        try:
            self.sock.sendto(osc_message(address, *args), self.addr)
        except OSError:
            pass  # nobody listening, or the network's down: tracking carries on

    def send_tracks(self, tracks):
        """One /person message per track (anything with .id and .pos in metres)."""
        for t in tracks:
            self.send("/person", int(t.id), float(t.pos[0]), float(t.pos[1]))
