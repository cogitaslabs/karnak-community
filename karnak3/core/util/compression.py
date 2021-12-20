from typing import Optional

import brotli
import base64
import zlib


def compress_base64(b: bytes, compression: str, **args) -> Optional[str]:
    if b is None:
        return None
    if compression in ['zlib', 'gz', 'gzip', 'brotli']:
        if compression == 'brotli':
            bytes_enc = brotli.compress(b, **args)
        elif compression in ['zlib', 'gz', 'gzip']:
            bytes_enc = zlib.compress(b, **args)
        else:
            assert False
        enc = base64.b64encode(bytes_enc).decode('ascii')
        return enc
    else:
        return b.decode()


def decompress_str_base64(s: str, compression: str) -> Optional[str]:
    if s is None:
        return None
    if compression in ['zlib', 'gz', 'gzip', 'brotli']:
        b = base64.b64decode(s)
        if compression == 'brotli':
            decompressed = brotli.decompress(b)
        elif compression in ['zlib', 'gz', 'gzip']:
            decompressed = zlib.decompress(b)
        else:
            assert False
        return decompressed.decode()
    else:
        return s
