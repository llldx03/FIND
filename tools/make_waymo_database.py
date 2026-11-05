"""Build the standard Waymo database for source training from prepared frames."""
import argparse
import json
import pickle
from pathlib import Path

import _init_path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.find.runtime import create_dataset
from pcdet.utils import common_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', required=True)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = create_dataset(cfg, True, common_utils.create_logger())
    preparation = json.loads((dataset.root_path / 'preprocess_summary.json').read_text())
    if not preparation['complete'] or len(dataset) != preparation['frames']:
        raise ValueError('Source training requires all selected training sequences')
    dataset.create_groundtruth_database(dataset.root_path / 'waymo_infos_train.pkl', dataset.root_path,
                                       used_classes=['Vehicle'], sampled_interval=2)
    path = dataset.root_path / 'pcdet_waymo_dbinfos_train_sampled_2.pkl'
    with path.open('rb') as stream:
        records = pickle.load(stream)
    count = len(records.get('Vehicle', []))
    if not count:
        raise RuntimeError('Empty Vehicle sampling database')
    print(json.dumps(dict(database=str(path), vehicle_objects=count, source_frames=len(dataset))), flush=True)


if __name__ == '__main__':
    main()
