# FullySparseFusion overview
- Purpose: research codebase for Fully Sparse Fusion (FSF), a fully sparse multi-modal 3D object detection framework using LiDAR and camera instances without dense BEV maps.
- Stack: Python, PyTorch, MMDetection3D-style registries/configs, conda environments.
- Main roots: `projects/mmdet3d_plugin/` for model code, `projects/configs/` for experiments, `tools/` for train/test scripts, `data/` and `ckpt/` for runtime assets.
- Common model lineage: `FSF` baseline with incremental variants like `FSF_HMamba`, `FSF_Occ`, `FSF_X`, and new detector variants that override only key fusion stages.
- Host context: Windows shell, often using the `FSF` conda environment for torch-based checks.