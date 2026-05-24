import os
from pathlib import Path

import pytest

from ._docs_examples import extract_markdown_samples, extract_rst_code_blocks

pytestmark = pytest.mark.skipif(
    os.environ.get('NUMPY_STL_RUN_DOCS_EXAMPLES') != '1',
    reason='Run docs example tests via tox -e docs-examples',
)


def test_extract_markdown_samples_tracks_language_and_line_numbers(speedups):
    source = Path('/tmp/README.md')
    text = (
        '# Heading\n'
        '\n'
        '```python\n'
        "print('hello')\n"
        '```\n'
        '\n'
        '```bash\n'
        'echo ok\n'
        '```\n'
    )

    samples = extract_markdown_samples(source, text)

    assert [sample.language for sample in samples] == ['python', 'bash']
    assert [sample.start_line for sample in samples] == [4, 8]
    assert samples[0].body == "print('hello')"
    assert samples[1].body == 'echo ok'


def test_extract_rst_code_blocks_preserves_blank_lines_and_options(speedups):
    source = Path('/tmp/guide.rst')
    text = (
        'Section\n'
        '=======\n'
        '\n'
        '.. code-block:: python\n'
        '   :caption: Example\n'
        '\n'
        '   import math\n'
        '\n'
        '   print(math.pi)\n'
        '\n'
        'Next section\n'
        '------------\n'
    )

    samples = extract_rst_code_blocks(source, text)

    assert len(samples) == 1
    assert samples[0].language == 'python'
    assert samples[0].start_line == 7
    assert samples[0].body == 'import math\n\nprint(math.pi)'
