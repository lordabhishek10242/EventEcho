"""Dependency-free Ed25519 signature verification (RFC 8032).

Only verification is needed: the grader signs receipts with a key whose public
half arrives as a JWK (``{"kty":"OKP","crv":"Ed25519","x":"..."}``).  This is
the classic reference implementation of the Edwards curve arithmetic plus
strict checks (point decoding, S < L, small-order rejection of the public key).
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493


def _inv(x: int) -> int:
    return pow(x, P - 2, P)


D = (-121665 * _inv(121666)) % P
I = pow(2, (P - 1) // 4, P)


def _x_recover(y: int) -> int:
    xx = (y * y - 1) * _inv(D * y * y + 1)
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P != 0:
        x = (x * I) % P
    if x % 2 != 0:
        x = P - x
    return x


_By = (4 * _inv(5)) % P
_Bx = _x_recover(_By)
BASE = (_Bx % P, _By % P, 1, (_Bx * _By) % P)
IDENTITY = (0, 1, 1, 0)


def _add(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % P
    b = ((y1 + x1) * (y2 + x2)) % P
    c = (t1 * 2 * D * t2) % P
    d = (z1 * 2 * z2) % P
    e, f, g, h = (b - a) % P, (d - c) % P, (d + c) % P, (b + a) % P
    return ((e * f) % P, (g * h) % P, (f * g) % P, (e * h) % P)


def _mul(point: tuple[int, int, int, int], scalar: int) -> tuple[int, int, int, int]:
    result = IDENTITY
    addend = point
    while scalar > 0:
        if scalar & 1:
            result = _add(result, addend)
        addend = _add(addend, addend)
        scalar >>= 1
    return result


def _equal(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]) -> bool:
    x1, y1, z1, _t1 = p1
    x2, y2, z2, _t2 = p2
    if z1 % P == 0 or z2 % P == 0:
        return z1 % P == z2 % P
    return (x1 * z2 - x2 * z1) % P == 0 and (y1 * z2 - y2 * z1) % P == 0


def _decode_point(data: bytes) -> tuple[int, int, int, int] | None:
    if len(data) != 32:
        return None
    value = int.from_bytes(data, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    if y >= P:
        return None
    x = _x_recover(y)
    if x == 0 and sign:
        return None
    if (x & 1) != sign:
        x = P - x
    if (-x * x + y * y - 1 - D * x * x * y * y) % P != 0:
        return None
    # reject small-order public keys / R values
    if _equal(_mul((x, y, 1, (x * y) % P), 8), IDENTITY):
        return None
    return (x, y, 1, (x * y) % P)


def b64decode_flexible(text: str) -> bytes:
    """Accept standard or URL-safe base64, with or without padding."""
    cleaned = text.strip().replace("-", "+").replace("_", "/").replace(" ", "")
    padding = (-len(cleaned)) % 4
    return base64.b64decode(cleaned + "=" * padding, validate=False)


def verify(signature: bytes, message: bytes, public_key: bytes) -> bool:
    if len(signature) != 64 or len(public_key) != 32:
        return False
    r_bytes = signature[:32]
    s_value = int.from_bytes(signature[32:], "little")
    if s_value >= L:
        return False
    point_r = _decode_point(r_bytes)
    point_a = _decode_point(public_key)
    if point_r is None or point_a is None:
        return False
    h = int.from_bytes(hashlib.sha512(r_bytes + public_key + message).digest(), "little") % L
    return _equal(_mul(BASE, s_value), _add(point_r, _mul(point_a, h)))


def public_key_from_jwk(jwk: dict[str, Any]) -> bytes | None:
    """Import an OKP/Ed25519 JWK; None when the key is not usable."""
    if not isinstance(jwk, dict):
        return None
    if str(jwk.get("kty", "")).upper() != "OKP":
        return None
    if str(jwk.get("crv", "")) not in {"Ed25519", "EdDSA"}:
        return None
    x = jwk.get("x")
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        raw = b64decode_flexible(x)
    except Exception:
        return None
    return raw if len(raw) == 32 else None


def verify_b64(signature_b64: str, message: bytes, public_key: bytes) -> bool:
    if not isinstance(signature_b64, str) or not signature_b64.strip():
        return False
    try:
        signature = b64decode_flexible(signature_b64)
    except Exception:
        return False
    return verify(signature, message, public_key)


# --------------------------------------------------------------------------- #
# Signing (used only by the test harness and the local receipt simulator)
# --------------------------------------------------------------------------- #

def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    x, y, z, _t = point
    zinv = _inv(z)
    x = (x * zinv) % P
    y = (y * zinv) % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def sign(seed: bytes, message: bytes) -> tuple[bytes, bytes]:
    """Return ``(signature, public_key)`` for a 32-byte RFC 8032 seed."""
    if len(seed) != 32:
        raise ValueError("seed must be 32 bytes")
    digest = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    public = _encode_point(_mul(BASE, scalar))
    prefix = digest[32:]
    nonce = int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % L
    encoded_r = _encode_point(_mul(BASE, nonce))
    challenge = int.from_bytes(hashlib.sha512(encoded_r + public + message).digest(), "little") % L
    s_value = (nonce + challenge * scalar) % L
    return encoded_r + s_value.to_bytes(32, "little"), public


def public_key_from_seed(seed: bytes) -> bytes:
    return sign(seed, b"")[1]
