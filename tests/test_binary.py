from stl import mesh


def test_ascii_like_binary(tmpdir, speedups):
    mesh.Mesh.from_file('tests/stl_binary/rear_case.stl', speedups=speedups)

    with open('tests/stl_binary/rear_case.stl', 'rb') as fh:
        mesh.Mesh.from_file('rear_case.stl', fh=fh, speedups=speedups)

