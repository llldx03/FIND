"""Prepare source mode statistics or run one chronological target pass with FIND."""
import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.find.adapter import detector_predictions, input_only, restore_rng
from pcdet.find.runtime import (atomic_torch_save, build_adapter, create_dataset, create_detector,
                                dataset_identity, graph_settings, jsonable, matches_config, prior_definition,
                                set_seed, stream_loader)
from pcdet.find.spectral import ModeMoments, object_spectra
from pcdet.find.voxelize import graph_points
from pcdet.models import load_data_to_gpu
from pcdet.utils import box_utils, common_utils


OUTPUT_DEFAULTS = dict(BOX_DIMENSION_SCALE=[1., 1., 1.], BOX_XY_OFFSET=[0., 0.], BOX_BOTTOM_OFFSET=0.,
                       SCORE_SIZE_PRIOR_WEIGHT=0., SCORE_SIZE_PRIOR_UPPER=[5.2, 2.05, 2.0],
                       SCORE_SIZE_PRIOR_MARGIN=[.6, .25, .35])


def detector_config(config):
    """Keep output-only settings out of detector construction and source-prior identity."""
    result = copy.deepcopy(config)
    for key in list(OUTPUT_DEFAULTS) + ['OUTPUT_SETTINGS_SOURCE']:
        if key in result.MODEL.POST_PROCESSING:
            result.MODEL.POST_PROCESSING.pop(key)
    return result


def output_settings(config):
    post = config.MODEL.POST_PROCESSING
    if not any(key in post for key in OUTPUT_DEFAULTS):
        return None
    settings = {key: copy.deepcopy(post.get(key, default)) for key, default in OUTPUT_DEFAULTS.items()}
    for key, length in [('BOX_DIMENSION_SCALE', 3), ('BOX_XY_OFFSET', 2),
                        ('SCORE_SIZE_PRIOR_UPPER', 3), ('SCORE_SIZE_PRIOR_MARGIN', 3)]:
        value = np.asarray(settings[key], dtype=np.float32)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(key + ' has an invalid shape or nonfinite value')
        if key != 'BOX_XY_OFFSET' and not (value > 0).all():
            raise ValueError(key + ' must be positive')
    if not np.isfinite(settings['BOX_BOTTOM_OFFSET']):
        raise ValueError('BOX_BOTTOM_OFFSET must be finite')
    if not np.isfinite(settings['SCORE_SIZE_PRIOR_WEIGHT']) or settings['SCORE_SIZE_PRIOR_WEIGHT'] < 0:
        raise ValueError('SCORE_SIZE_PRIOR_WEIGHT must be finite and nonnegative')
    return settings


def postprocess_predictions(predictions, dataset, settings, start=0):
    """Apply configured Car output settings after KITTI conversion and FOV filtering."""
    if settings is None:
        return predictions
    scale = np.asarray(settings['BOX_DIMENSION_SCALE'], dtype=np.float32)
    upper = np.asarray(settings['SCORE_SIZE_PRIOR_UPPER'], dtype=np.float32)
    margin = np.asarray(settings['SCORE_SIZE_PRIOR_MARGIN'], dtype=np.float32)
    weight = settings['SCORE_SIZE_PRIOR_WEIGHT']
    processed = []
    for index, original in enumerate(predictions, start):
        info = dataset.kitti_infos[index]
        frame = str(info['point_cloud']['lidar_idx'])
        if str(original['frame_id']) != frame:
            raise ValueError('Prediction frame order does not match the target stream')
        result = copy.deepcopy(original)
        boxes = result['boxes_lidar']
        if len(boxes):
            selected = result['name'] == 'Car'
            old_height = boxes[selected, 5].copy()
            boxes[selected, :2] += np.asarray(settings['BOX_XY_OFFSET'], dtype=boxes.dtype)
            boxes[selected, 3:6] *= scale
            boxes[selected, 2] += .5 * (boxes[selected, 5] - old_height) + settings['BOX_BOTTOM_OFFSET']
            if weight:
                excess = np.maximum(boxes[selected, 3:6] - upper, 0.) / margin
                result['score'][selected] *= np.exp(-weight * np.square(excess).sum(axis=1))
            calib = dataset.get_calib(frame)
            camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes, calib)
            result['bbox'] = box_utils.boxes3d_kitti_camera_to_imageboxes(camera, calib, info['image']['image_shape'])
            result['dimensions'] = camera[:, 3:6]
            result['location'] = camera[:, :3]
            result['rotation_y'] = camera[:, 6]
            result['alpha'] = -np.arctan2(-boxes[:, 1], boxes[:, 0]) + camera[:, 6]
            assert np.isfinite(boxes).all()
        processed.append(result)
    return processed


def prepare_prior(args, logger):
    model_config = detector_config(cfg)
    dataset = create_dataset(cfg, source=True, logger=logger)
    model = create_detector(model_config, dataset, args.checkpoint, logger).eval()
    moments = ModeMoments(cfg.FIND.FILTER_BINS).cuda()
    identity = dict(dataset=dataset_identity(dataset), definition=prior_definition(model_config), seed=args.seed)
    samples, graphs, max_graph = 0, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu')
        if saved['format_version'] != 1 or not matches_config(saved['identity'], identity):
            raise ValueError('Source prior resume configuration does not match')
        moments.load_state_dict(saved['moments'])
        samples, graphs, max_graph = saved['samples_seen'], saved['graphs'], saved['largest_graph']
        if not 0 <= samples <= len(dataset):
            raise ValueError('Invalid source prior progress')
        restore_rng(saved['rng'])
    from pcdet.find.adapter import capture_rng

    def save():
        payload = dict(format_version=1, identity=identity, moments={k: v.cpu() for k, v in moments.state_dict().items()},
                       samples_seen=samples, graphs=graphs, largest_graph=max_graph,
                       complete=samples == len(dataset), rng=capture_rng())
        atomic_torch_save(payload, args.output)
        summary = {key: value for key, value in payload.items() if key not in ('moments', 'rng')}
        summary.update(count=moments.count.tolist(), sum=moments.sum.tolist(), sum_sq=moments.sum_sq.tolist(),
                       stability=moments.stability(cfg.FIND.TAU).tolist())
        Path(str(args.output) + '.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    for index, batch in enumerate(stream_loader(dataset, args.batch_size, args.workers, args.seed, samples)):
        load_data_to_gpu(batch)
        batch = input_only(batch)
        with torch.no_grad():
            predictions = detector_predictions(model, dict(batch))
            nodes = graph_points(batch, cfg.FIND.get('GRAPH_REPRESENTATION', 'points'))
            spectra = object_spectra(nodes, predictions, retain_vectors=False, **graph_settings(cfg))
            for spectrum in spectra:
                moments.add(spectrum.eigenvalues, spectrum.coefficients.square().sum(dim=1))
                max_graph = max(max_graph, len(spectrum.indices))
        samples += batch['batch_size']
        graphs += len(spectra)
        logger.info('source samples=%d/%d graphs=%d modes=%d largest=%d', samples, len(dataset), graphs,
                    int(moments.count.sum()), max_graph)
        if (index + 1) % args.save_every == 0:
            save()
        if args.max_batches and index + 1 >= args.max_batches:
            break
    save()


def online(args, logger):
    model_config = detector_config(cfg)
    settings = output_settings(cfg)
    prior = torch.load(args.prior, map_location='cpu')
    if prior['format_version'] != 1 or not matches_config(prior['identity']['definition'], prior_definition(model_config)):
        raise ValueError('Source prior configuration or mode definition does not match')
    if not prior['complete'] and not args.allow_partial_prior:
        raise ValueError('Source prior is incomplete')
    dataset = create_dataset(cfg, source=False, logger=logger)
    full_steps = math.ceil(len(dataset) / args.batch_size)
    adapter = build_adapter(model_config, dataset, args.checkpoint, prior, logger, full_steps)
    resume_config = dict(config=jsonable(cfg), batch_size=args.batch_size, seed=args.seed,
                         stream=dataset_identity(dataset), total_steps=full_steps, prior_complete=prior['complete'])
    if settings is not None:
        resume_config['output_postprocessing'] = dict(settings=jsonable(settings))
    annotations, raw_annotations, history = [], [], []
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        saved_config = state.get('resume_config', state.get('provenance'))
        if not matches_config(saved_config, resume_config):
            raise ValueError('Resume configuration or target stream does not match')
        adapter.load_state_dict(state)
        annotations, history = state['annotations'], state['history']
        if adapter.samples_seen > len(dataset) or adapter.step_index > full_steps:
            raise ValueError('Checkpoint progress exceeds the target stream')
        if len(annotations) != adapter.samples_seen:
            raise ValueError('Saved predictions do not match stream position')
        if settings is not None:
            raw_annotations = state['raw_annotations']
            if len(raw_annotations) != adapter.samples_seen:
                raise ValueError('Saved raw predictions do not match stream position')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / 'latest.pth').exists():
        raise FileExistsError('Run already exists')

    def save():
        state = adapter.state_dict()
        state.update(resume_config=copy.deepcopy(resume_config), annotations=annotations, history=history)
        if settings is not None:
            state['raw_annotations'] = raw_annotations
        atomic_torch_save(state, output / 'latest.pth')
        with (output / 'predictions.pkl').open('wb') as stream:
            pickle.dump(annotations, stream)
        if settings is not None:
            with (output / 'predictions_raw.pkl').open('wb') as stream:
                pickle.dump(raw_annotations, stream)
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

    for index, batch in enumerate(stream_loader(dataset, args.batch_size, args.workers, args.seed, adapter.samples_seen)):
        start = time.perf_counter()
        load_data_to_gpu(batch)
        predictions, metrics = adapter.step(batch)
        batch_annotations = dataset.generate_prediction_dicts(input_only(batch), copy.deepcopy(predictions), dataset.class_names)
        if settings is not None:
            raw_annotations.extend(batch_annotations)
        annotations.extend(postprocess_predictions(batch_annotations, dataset, settings, start=len(annotations)))
        metrics['seconds'] = time.perf_counter() - start
        history.append(metrics)
        logger.info(json.dumps(metrics))
        if adapter.step_index % args.save_every == 0:
            save()
        if args.max_batches and index + 1 >= args.max_batches:
            break
    save()
    summary = dict(samples_seen=adapter.samples_seen, dataset_samples=len(dataset), steps=adapter.step_index,
                   complete=adapter.samples_seen == len(dataset), prior_complete=prior['complete'],
                   checkpoint=str(output / 'latest.pth'))
    if args.evaluate:
        if adapter.samples_seen != len(dataset):
            raise ValueError('--evaluate requires predictions for the entire target stream')
        result, metrics = dataset.evaluation(annotations, dataset.class_names, eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                                            output_path=output / 'evaluation')
        (output / 'evaluation.txt').write_text(result, encoding='utf-8')
        summary['metrics'] = jsonable(metrics)
        logger.info(result)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['source-prior', 'online'])
    parser.add_argument('--cfg-file', default='cfgs/tta_n2k_models/secondiou/find.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prior')
    parser.add_argument('--resume')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--save-every', type=int, default=32)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--allow-partial-prior', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--set', dest='overrides', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(args.overrides, cfg)
    args.batch_size = args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    if args.batch_size < 1 or args.save_every < 1 or args.max_batches < 0:
        parser.error('Invalid batch size, save interval, or batch limit')
    if args.mode == 'online' and not args.prior:
        parser.error('--prior is required for online adaptation')
    if cfg.FIND.FILTER_ORDER != 3:
        parser.error('Only cubic FIND filters are configured')
    set_seed(args.seed)
    logger = common_utils.create_logger()
    if args.mode == 'source-prior':
        prepare_prior(args, logger)
    else:
        online(args, logger)


if __name__ == '__main__':
    main()
