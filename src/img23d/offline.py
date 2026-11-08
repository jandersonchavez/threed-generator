"""Bloqueia conexões Python externas durante a geração, preservando loopback."""

import ipaddress
import sys

_installed = False


def enforce_offline():
    global _installed
    if _installed:
        return

    def local(host):
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def audit(event, args):
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and not local(address[0]):
                raise OSError("Modo offline: conexão externa bloqueada.")
        elif event == "socket.getaddrinfo" and not local(args[0]):
            raise OSError("Modo offline: consulta DNS externa bloqueada.")

    sys.addaudithook(audit)
    _installed = True
