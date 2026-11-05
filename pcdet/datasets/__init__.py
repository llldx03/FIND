"""Dataset construction; SDKs are loaded only for the selected dataset."""
import copy
import importlib
import random
from functools import partial


DATASETS = {
    'KittiDataset': '.kitti.kitti_dataset',
    'NuScenesDataset': '.nuscenes.nuscenes_dataset',
    'WaymoDataset': '.waymo.waymo_dataset',
}


def initialize_worker(worker_id, config):
    import numpy as np
    import torch
    from pcdet.config import cfg
    cfg.clear()
    cfg.update(config)
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):
    from torch.utils.data import DataLoader, DistributedSampler
    from pcdet.config import cfg, resolve_data_path

    module = importlib.import_module(DATASETS[dataset_cfg.DATASET], __name__)
    dataset = getattr(module, dataset_cfg.DATASET)(
        dataset_cfg=dataset_cfg, class_names=class_names,
        root_path=resolve_data_path(root_path or dataset_cfg.DATA_PATH), training=training, logger=logger,
    )
    if merge_all_iters_to_one_epoch:
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
    sampler = DistributedSampler(dataset, shuffle=training) if dist else None
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
        shuffle=training and sampler is None, sampler=sampler, drop_last=False,
        collate_fn=dataset.collate_batch, persistent_workers=workers > 0,
        worker_init_fn=partial(initialize_worker, config=copy.deepcopy(cfg)),
    )
    return dataset, loader, sampler
