"""A single explicit failure boundary for implementations awaiting migration."""


class RuntimeNotAvailableError(NotImplementedError):
    """Raised instead of simulating an unavailable WISE runtime component."""

    def __init__(self, component: str) -> None:
        super().__init__(
            f"{component} is an interface in this framework release. "
            "The WISE runtime implementation and required experiment assets "
            "have not been migrated. Use `python -m wise --describe` to inspect "
            "the architecture; this release does not run paper evaluations."
        )
