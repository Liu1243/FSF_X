import os
import shutil
import glob
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class SaveBestAsOptHook(Hook):
    """Hook to save/symlink the best checkpoint as 'opt.pth'.

    Args:
        metric (str): The metric used to determine the best checkpoint.
            Default: 'pts_bbox_NuScenes/NDS'.
    """

    def __init__(self, metric='pts_bbox_NuScenes/NDS'):
        self.metric = metric.replace('/', '_') # mmcv replaces / with _ in filenames

    def after_train_epoch(self, runner):
        # We run this after every epoch. If a 'best_*.pth' exists, we link it to 'opt.pth'.
        # Note: mmcv's EvalHook saves the best checkpoint as 'best_<metric>.pth'
        # in the runner.work_dir.
        
        best_ckpt_path = os.path.join(runner.work_dir, f'best_{self.metric}.pth')
        opt_path = os.path.join(runner.work_dir, 'opt.pth')
        
        if os.path.exists(best_ckpt_path):
            # Use copy instead of symlink to ensure the file is portable/persistent
            shutil.copy(best_ckpt_path, opt_path)
            runner.logger.info(f'Best checkpoint copied to {opt_path}')
