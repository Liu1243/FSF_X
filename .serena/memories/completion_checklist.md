# Completion checklist
- Run focused tests in the `FSF` conda environment for any new torch-only modules.
- Run `python -m py_compile` on newly added Python files for syntax validation.
- Check `git status --short` to ensure only intended files changed.
- If the task adds a new experiment variant, make sure the detector is exported in `projects/mmdet3d_plugin/models/detectors/__init__.py` and a config file exists under `projects/configs/`.
- Report clearly when full training/inference validation could not be run due missing dependencies, data, or environment gaps.