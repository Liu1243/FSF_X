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
        class_groups=[
            [2, 3, 4, 20, 21, 22, 23, 24],
            [5, 6, 7, 8, 9, 10],
            [15, 19],
        ],
        class_group_min_tokens=[2, 2, 1],
        num_rotations=2,
        window_size=32,
        use_fast_path=False,
        use_extended_reliability=True,
        small_object_class_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19, 20, 21, 22, 23, 24],
        small_object_residual_scale=0.1,
    ),
)
