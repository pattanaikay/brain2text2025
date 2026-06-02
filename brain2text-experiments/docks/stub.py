"""
docks/stub.py
-------------
Shared machinery for SKELETON experiments — code that wires up and shape-checks
but must NOT be mistaken for a runnable result.

The contract (ADHD convergence 2026-05-30, frame: 3am-on-call + competitor):
  - A stub stage builds a real Module of the correct out_shape, so
    Stack.from_spec wiring is provably correct.
  - Its forward() raises StubNotImplemented unless the owning spec explicitly
    sets `allow_stub_forward: true`, so training is provably blocked.
  - A tripwire test asserts the stub IS still a stub; it fails loudly the day
    someone half-implements forward() without flipping the registry state.

This is one symbol + one flag so every present and future skeleton inherits the
same provable-stub behaviour, and the tripwire only has to check one thing.
"""

from __future__ import annotations


class StubNotImplemented(NotImplementedError):
    """Raised when a skeleton stage's forward() runs without explicit opt-in."""


def guard_stub_forward(module, *, hint: str = ""):
    """
    Call at the top of a skeleton Module.forward(). Raises StubNotImplemented
    unless `module.allow_stub_forward` is truthy.

    Usage:
        def forward(self, x):
            guard_stub_forward(self, hint="episodic read head")
            ...
    """
    if not getattr(module, "allow_stub_forward", False):
        name = type(module).__name__
        raise StubNotImplemented(
            f"{name}.forward() is a SKELETON stub{(' — ' + hint) if hint else ''}.\n"
            "Set `allow_stub_forward: true` in the spec to exercise the wiring,\n"
            "or implement the real forward and flip the registry entry's "
            "state from 'skeleton' to 'partial'/'live'."
        )
