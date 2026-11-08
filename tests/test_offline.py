import subprocess
import sys


def test_external_network_blocked_and_loopback_resolves():
    # O audit hook persiste: teste em processo isolado para não afetar outros.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket
from img23d.offline import enforce_offline
enforce_offline()
assert socket.getaddrinfo('127.0.0.1', 7860)
try:
    socket.getaddrinfo('example.com', 443)
except OSError as exc:
    assert 'offline' in str(exc)
else:
    raise AssertionError('DNS externo permitido')
with socket.socket() as connection:
    try:
        connection.connect(('1.1.1.1', 443))
    except OSError as exc:
        assert 'offline' in str(exc)
    else:
        raise AssertionError('Conexão externa permitida')
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
