def b(
    s: 'str | bytes',
    encoding: str = 'ascii',
    errors: str = 'replace',
) -> bytes:  # pragma: no cover
    if isinstance(s, str):
        return bytes(s, encoding, errors)
    else:
        return s
