"""Explicit limits on live numerical workspace, excluding inputs and outputs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionOptions:
    """Maximum estimated intermediate bytes; allocator/compiler overhead is separate."""

    workspace_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        """Require enough room for one point and one source patch."""
        if (
            not isinstance(self.workspace_bytes, int)
            or isinstance(self.workspace_bytes, bool)
            or self.workspace_bytes < 4096
        ):
            raise ValueError("workspace_bytes must be an integer of at least 4096")
