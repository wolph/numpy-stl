def b(
    s: 'str | bytes',
    encoding: str = 'ascii',
    errors: str = 'replace',
) -> bytes:
    """Encode a string to bytes, passing bytes through.

    Args:
        s: String or bytes input.
        encoding: Encoding to use. Defaults to ``'ascii'``.
        errors: Error handling strategy. Defaults to
            ``'replace'``.

    Returns:
        Encoded bytes.
    """
    if isinstance(s, str):
        return bytes(s, encoding, errors)
    else:
        return s
