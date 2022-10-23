def b(s, encoding='ascii', errors='replace'):  # pragma: no cover
    if isinstance(s, str):
        return bytes(s, encoding, errors)
    else:
        return s
    # return bytes(s, encoding, errors)
