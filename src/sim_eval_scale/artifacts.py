"""Publish complete immutable snapshots, including hot >1MiB JSON, by atomic rename."""

import hashlib
import os
import tempfile
from pathlib import Path


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise ValueError("do not follow source cache symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    h = hashlib.sha256()
    written = 0
    fd, name = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".pending", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
            for data in iter(lambda: src.read(1 << 20), b""):
                dst.write(data)
                h.update(data)
                written += len(data)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or written != before.st_size:
            return {"published": False, "reason": "source_changed_retry_next_pass"}
        # Readback checks the storage destination bytes before publication.
        readback = hashlib.sha256()
        with temporary.open("rb") as stream:
            for data in iter(lambda: stream.read(1 << 20), b""):
                readback.update(data)
        if readback.digest() != h.digest():
            raise IOError("destination SHA256 mismatch")
        temporary.replace(destination)
        return {"published": True, "bytes": written, "sha256": h.hexdigest()}
    finally:
        temporary.unlink(missing_ok=True)
