"""
stages/memory/
--------------
Optional memory stage kind, inserted between encoder and projector by
Stack.from_spec when a spec declares a `memory:` block. Existing specs have no
`memory:` key and skip this stage entirely, so the 25 live experiments are
unaffected.

Currently houses Track H's ZenBrain episodic-memory SKELETON.
"""
