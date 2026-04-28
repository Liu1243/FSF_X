# Style and conventions
- Python code uses registry-based model definitions (`@DETECTORS.register_module()` etc.) and constructor-driven config injection.
- Existing code heavily uses inline Chinese comments/docstrings to explain model stages; matching that style is acceptable.
- Model variants typically inherit from `FSF` and override only the necessary stage such as `combine_frustum_and_fsd`.
- Config files are plain Python dicts and usually keep most settings from a parent config, overriding only the changed detector type or module config.
- New utility modules should stay PyTorch-only when possible so they can be unit-tested without the full MMDetection stack.
- Avoid reverting unrelated user changes; this repo may be in a dirty worktree.