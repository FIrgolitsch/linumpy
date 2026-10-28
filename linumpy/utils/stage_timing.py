"""Env-gated stage timers. Reads ``LINUMPY_STAGE_TIMING`` on each call."""


class span:  # noqa: N801
    """Context manager that does not record a duration."""

    def __init__(self, stage: str, **meta: object) -> None:
        """Store the stage name without starting a clock.

        Parameters
        ----------
        stage : str
            Stage name the caller asked to time.
        **meta : object
            Extra fields the caller attached to the stage.
        """
        self.stage = stage
        self.meta = meta

    def __enter__(self) -> span:
        """Return this context without reading the clock."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave the wrapped value unchanged and record nothing."""
        return None


def flush() -> None:
    """Do not write a timing file."""
    return None
