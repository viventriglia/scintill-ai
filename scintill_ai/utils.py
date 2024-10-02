import sys
from typing import Iterable, Iterator


def progressbar(iterable: Iterable, prefix: str = "", size: int = 50) -> Iterator:
    count = len(iterable)

    def show(j: int) -> None:
        x = int(size * j / count)
        sys.stdout.write(
            "%s[%s%s] %i%%\r"
            % (prefix, "#" * x, "." * (size - x), int(100 * j / count))
        )
        sys.stdout.flush()

    show(0)
    for i, item in enumerate(iterable):
        yield item
        show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()
