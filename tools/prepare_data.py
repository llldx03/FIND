"""Prepare KITTI or nuScenes metadata and the source sampling database."""
import argparse
import pickle
from pathlib import Path

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file, resolve_data_path
from pcdet.utils import common_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['kitti', 'nuscenes'])
    parser.add_argument('--data-path', help='Dataset root; overrides DATA_PATH in the config')
    parser.add_argument('--cfg-file')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file or f'cfgs/dataset_configs/{args.dataset}_dataset.yaml', cfg)
    cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
    root = resolve_data_path(args.data_path or cfg.DATA_PATH)
    logger = common_utils.create_logger()
    if args.dataset == 'kitti':
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
        dataset = KittiDataset(cfg, ['Car'], root_path=root, training=False, logger=logger)
        combined = []
        for split in ['train', 'val']:
            dataset.set_split(split)
            infos = dataset.get_infos(num_workers=args.workers, has_label=True, count_inside_pts=True)
            with (root / f'kitti_infos_{split}.pkl').open('wb') as stream:
                pickle.dump(infos, stream)
            combined.extend(infos)
        with (root / 'kitti_infos_trainval.pkl').open('wb') as stream:
            pickle.dump(combined, stream)
        dataset.set_split('train')
        dataset.create_groundtruth_database(root / 'kitti_infos_train.pkl', used_classes=['Car'], split='train')
    else:
        from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset, create_nuscenes_info
        create_nuscenes_info(cfg.VERSION, root, root, max_sweeps=cfg.MAX_SWEEPS)
        cfg.BALANCED_RESAMPLING = False
        dataset = NuScenesDataset(cfg, ['car'], root_path=root, training=True, logger=logger)
        dataset.create_groundtruth_database(used_classes=['car'], max_sweeps=cfg.MAX_SWEEPS)


if __name__ == '__main__':
    main()
