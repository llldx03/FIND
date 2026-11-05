"""Evaluate a source SECOND-IoU checkpoint on the configured KITTI target split."""
import argparse
import copy
import json
import pickle
from pathlib import Path

import _init_path
import torch

from find import detector_config, output_settings, postprocess_predictions
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.find.adapter import detector_predictions, input_only
from pcdet.find.runtime import create_dataset, create_detector, jsonable, set_seed, stream_loader
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-file', required=True, help='One of the three task find.yaml files')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--set', dest='overrides', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Invalid batch size or worker count')
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(args.overrides, cfg)
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(str(output / 'test.log'))
    dataset = create_dataset(cfg, source=False, logger=logger)
    model = create_detector(detector_config(cfg), dataset, args.checkpoint, logger).eval()
    settings = output_settings(cfg)
    predictions, raw = [], []
    with torch.no_grad():
        for batch in stream_loader(dataset, args.batch_size, args.workers, args.seed):
            load_data_to_gpu(batch)
            predicted = detector_predictions(model, input_only(batch))
            annotations = dataset.generate_prediction_dicts(input_only(batch), copy.deepcopy(predicted), dataset.class_names)
            raw.extend(annotations)
            predictions.extend(postprocess_predictions(annotations, dataset, settings, start=len(predictions)))
    for name, value in [('predictions.pkl', predictions), ('predictions_raw.pkl', raw)]:
        with (output / name).open('wb') as stream:
            pickle.dump(value, stream)
    result, metrics = dataset.evaluation(predictions, dataset.class_names, eval_metric='kitti', output_path=output)
    (output / 'evaluation.txt').write_text(result, encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(dict(
        samples=len(predictions), metrics=jsonable(metrics), output_settings=settings,
    ), indent=2), encoding='utf-8')
    logger.info(result)


if __name__ == '__main__':
    main()
