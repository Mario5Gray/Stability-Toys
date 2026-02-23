"""Shared HTTP utilities for server routes."""

from urllib import request as urlrequest


def post_bytes(url: str, body: bytes, content_type: str) -> int:
    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        return resp.status
