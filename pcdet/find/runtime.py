"""Dataset and checkpoint helpers shared by FIND preparation and online execution."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def matches_config(saved, expected):
    if isinstance(expected, dict):
        return isinstance(saved, dict) and all(
            key in saved and matches_config(saved[key], value) for key, value in expected.items()
        )
    return saved == expected


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_dataset(config, source, logger):
    from pcdet.datasets import build_dataloader
    data_config = copy.deepcopy(config.DATA_CONFIG if source else config.DATA_CONFIG_TAR)
    data_config.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
    if source:
        data_config.INFO_PATH['test'] = data_config.INFO_PATH['train']
        data_config.DATA_SPLIT['test'] = data_config.DATA_SPLIT['train']
    for processor in data_config.DATA_PROCESSOR:
        if processor.NAME == 'shuffle_points':
            processor.SHUFFLE_ENABLED['test'] = False
    class_names = config.CLASS_NAMES if source else data_config.CLASS_NAMES
    dataset, _, _ = build_dataloader(data_config, class_names, batch_size=1, dist=False,
                                     workers=0, logger=logger, training=False)
    if not len(dataset):
        raise ValueError('Requested dataset is empty')
    return dataset


def stream_loader(dataset, batch_size, workers, seed, start=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, range(start, len(dataset))), batch_size=batch_size,
                      shuffle=False, num_workers=workers, pin_memory=True, drop_last=False,
                      collate_fn=dataset.collate_batch, generator=generator)


def dataset_identity(dataset):
    return dict(config=jsonable(dataset.dataset_cfg), samples=len(dataset))


def create_detector(config, dataset, checkpoint, logger):
    from pcdet.models import build_network
    detector_dataset = copy.copy(dataset)
    detector_dataset.class_names = list(config.CLASS_NAMES)
    model = build_network(model_cfg=copy.deepcopy(config.MODEL), num_class=len(config.CLASS_NAMES), dataset=detector_dataset)
    model.load_params_from_file(filename=str(checkpoint), logger=logger, to_cpu=True)
    return model.cuda()


def graph_settings(config):
    return dict(neighbors=config.FIND.NUM_NEIGHBORS, sigma=config.FIND.RBF_SIGMA,
                score_threshold=config.FIND.GRAPH_SCORE_THRESHOLD,
                backend=config.FIND.get('GRAPH_BACKEND', 'dense'))


def prior_definition(config):
    return dict(model=jsonable(config.MODEL), class_names=list(config.CLASS_NAMES),
                source_data_config=jsonable(config.DATA_CONFIG), graph=graph_settings(config),
                bins=config.FIND.FILTER_BINS,
                graph_representation=config.FIND.get('GRAPH_REPRESENTATION', 'points'),
                energy_observation='squared_norm_of_centered_object_spectral_mode')


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def build_adapter(config, dataset, checkpoint, prior, logger, total_steps):
    from tools.train_utils.optimization import build_optimizer, build_scheduler
    from .adapter import FINDAdapter
    from .spectral import ModeMoments, SpectralViews
    from .voxelize import DifferentiableVoxelizer
    if config.FIND.FILTER_ORDER != 3 or config.OPTIMIZATION.OPTIMIZER != 'adam_onecycle':
        raise ValueError('This configuration requires cubic filters and the MOS Adam one-cycle optimizer')
    if config.FIND.get('GRAPH_REPRESENTATION', 'points') == 'voxel' and (
            config.MODEL.VFE.NAME != 'MeanVFE' or config.MODEL.NAME != 'SECONDNetIoU'):
        raise ValueError('Voxel FIND currently uses the existing MeanVFE/SECONDNetIoU feature interface')
    moments = ModeMoments(config.FIND.FILTER_BINS)
    moments.load_state_dict(prior['moments'], strict=True)
    views = SpectralViews(moments, tau=config.FIND.TAU, eta=config.FIND.ETA_INIT,
                          threshold_rate=config.FIND.THRESHOLD_RATE).cuda()
    student = create_detector(config, dataset, checkpoint, logger)
    teacher = copy.deepcopy(student)
    optimizer = build_optimizer(student, config.OPTIMIZATION)
    scheduler, _ = build_scheduler(optimizer, total_iters_each_epoch=max(3, total_steps),
                                    total_epochs=1, last_epoch=-1, optim_cfg=config.OPTIMIZATION)
    filter_optimizer = torch.optim.Adam(views.parameters(), lr=config.OPTIMIZATION.LR, betas=(0.9, 0.99))
    return FINDAdapter(student, teacher, views, DifferentiableVoxelizer.from_dataset(dataset),
                       optimizer, filter_optimizer, scheduler, config.FIND,
                       config.SELF_TRAIN.SCORE_THRESH, config.SELF_TRAIN.NEG_THRESH)
