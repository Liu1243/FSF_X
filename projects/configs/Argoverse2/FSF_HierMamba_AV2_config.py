_base_ = [
    './FSF_HMamba_AV2_config.py',
]

model = dict(
    type='FSF_HierMamba',
    hiermamba_cfg=dict(
        d_model=1024,
        d_state=16,
        expand_factor=2,
        dt_rank='auto',
        conv_kernel=4,
        keep_ratio=0.75,
        min_tokens=16,
        min_per_modality=1,
        num_rotations=2,
        window_size=32,
        use_fast_path=False,
    ),
)
