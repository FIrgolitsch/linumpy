"""Progress lines of a long step, for the console of the workflow.

A step that runs as one task shows in Nextflow as a single line until it
ends. With ``debug true`` on the process, Nextflow prints what the task
writes to standard output while it runs, so a few lines there are the
progress of the step. Progress bars and the log go to standard error and
stay in the task's own log.
"""

import time


def report(step: str, message: str) -> None:
    """Print one progress line of ``step`` to standard output."""
    print(f"[{step}] {message}", flush=True)


class Counter:
    """Report a loop about ten times over its length: ``[step] what done/total``.

    Parameters
    ----------
    step : str
        Name of the step, as the workflow calls it.
    what : str
        What is counted, for example ``"slabs stacked"``.
    total : int
        Number of items.
    """

    def __init__(self, step: str, what: str, total: int) -> None:
        """Remember the labels and start the clock."""
        self.step = step
        self.what = what
        self.total = max(1, int(total))
        self.done = 0
        self._next = 0
        self._start = time.perf_counter()

    def update(self, n: int = 1) -> None:
        """Count ``n`` more items and print a line when another tenth is done."""
        self.done += n
        tenth = (10 * self.done) // self.total
        if tenth > self._next or self.done >= self.total:
            self._next = tenth
            elapsed = time.perf_counter() - self._start
            report(self.step, f"{self.what} {min(self.done, self.total)}/{self.total} ({elapsed:.0f} s)")
