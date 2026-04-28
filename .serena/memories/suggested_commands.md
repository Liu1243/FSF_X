# Suggested commands
- Activate env in cmd: `call C:\Users\1\miniconda3\Scripts\activate.bat FSF`
- Train nuScenes baseline/HMamba variants: `python tools/train.py projects/configs/nuScenes/<config>.py --work-dir work_dirs/nuScenes/<run_name>`
- Test: `python tools/test.py projects/configs/nuScenes/<config>.py <checkpoint> --eval bbox`
- Shell listing on Windows PowerShell: `Get-ChildItem`, recursive file scan: `Get-ChildItem -Recurse`
- Text search fallback when `rg` is unavailable: `git grep -n "pattern"`
- Syntax check: `python -m py_compile <file1> <file2> ...`
- Unit test for the new HierMamba module: `python -m unittest tests/test_hierarchical_hmamba.py`
- Dataset mask preprocessing scripts: `tools/mask_tools/save_mask_nusc.sh`, `tools/mask_tools/save_mask_argo2.sh`
- Git status: `git status --short`