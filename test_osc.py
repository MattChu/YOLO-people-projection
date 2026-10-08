#!/usr/bin/env python3
"""
test_osc.py - check osc.py's messages decode as standard OSC, and arrive over UDP.

  python test_osc.py
"""

import socket
import struct
from types import SimpleNamespace

from osc import OscSender, osc_message


def read_string(data, i):
    end = data.index(b"\0", i)
    return data[i:end].decode(), (end // 4 + 1) * 4


def decode(data):
    """A minimal OSC decoder, written from the spec, independent of osc.py."""
    assert len(data) % 4 == 0, "OSC packets are a multiple of 4 bytes"
    address, i = read_string(data, 0)
    tags, i = read_string(data, i)
    assert tags.startswith(",")
    args = []
    for t in tags[1:]:
        if t == "i":
            args.append(struct.unpack(">i", data[i:i + 4])[0]); i += 4
        elif t == "f":
            args.append(struct.unpack(">f", data[i:i + 4])[0]); i += 4
        elif t == "s":
            s, i = read_string(data, i); args.append(s)
    assert i == len(data)
    return address, args


def test_person_message():
    address, (pid, x, y) = decode(osc_message("/person", 7, 1.25, 0.5))
    assert (address, pid, x, y) == ("/person", 7, 1.25, 0.5)


def test_padding_at_every_length():
    for n in range(1, 9):
        address, args = decode(osc_message("/" + "a" * n, "b" * n, -3, 2.5))
        assert address == "/" + "a" * n and args == ["b" * n, -3, 2.5]


def test_known_bytes():
    # "/person" (7 chars + 1 null), ",iff" (4 chars + 4 nulls), then big-endian values
    assert osc_message("/person", 1, 0.0, 1.0) == (b"/person\0" + b",iff\0\0\0\0" + b"\0\0\0\1" +
                                                    b"\0\0\0\0" + b"\x3f\x80\0\0")


def test_sends_over_udp():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    sender = OscSender("127.0.0.1", rx.getsockname()[1])
    sender.send_tracks([SimpleNamespace(id=3, pos=(0.4, 1.1)), SimpleNamespace(id=4, pos=(2.0, 0.2))])
    got = [decode(rx.recv(1024)) for _ in range(2)]
    assert [(a, p[0]) for a, p in got] == [("/person", 3), ("/person", 4)]
    assert abs(got[0][1][1] - 0.4) < 1e-6 and abs(got[1][1][2] - 0.2) < 1e-6


def test_nobody_listening_is_fine():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # now nothing listens on that port
    sender = OscSender("127.0.0.1", port)
    for _ in range(3):
        sender.send("/person", 1, 0.0, 0.0)  # mustn't raise


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
