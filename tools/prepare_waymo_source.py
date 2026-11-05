"""Prepare the source-training frames with the repository's Waymo converter."""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import pickle
import shutil
import time
from pathlib import Path

import _init_path


def worker_init():
    import numpy as np
    for name, value in [('bool', bool), ('int', int), ('float', float),
                        ('complex', complex), ('object', object), ('str', str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from waymo_open_dataset.utils import frame_utils
    original_parser = frame_utils.parse_range_image_and_camera_projection

    def parse_compatible(frame):
        result = original_parser(frame)
        if len(result) == 4:
            return result[0], result[1], result[3]
        return result

    frame_utils.parse_range_image_and_camera_projection = parse_compatible


def process_one(task):
    from pcdet.datasets.waymo import waymo_utils
    source, destination, interval = task
    infos = waymo_utils.process_single_sequence(Path(source), Path(destination), interval, has_label=True)
    if not infos:
        raise RuntimeError('No frames produced: ' + source)
    return Path(source).stem, len(infos)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--max-sequences', type=int, default=0)
    args = parser.parse_args()
    raw_root, output = Path(args.raw_root).resolve(), Path(args.output).resolve()
    if args.workers < 1 or args.sample_interval < 1 or args.max_sequences < 0:
        parser.error('Invalid worker count, sample interval, or sequence limit')
    train_split = raw_root / 'ImageSets/train.txt'
    sequence_names = train_split.read_text().splitlines()
    sources = []
    for name in sequence_names:
        path = raw_root / 'raw_data' / name.strip()
        if not path.exists():
            path = path.with_name(path.name.replace('.tfrecord', '_with_camera_labels.tfrecord'))
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(path)
    expected_sequences = len(sources)
    if args.max_sequences:
        sources = sources[:args.max_sequences]
    (output / 'ImageSets').mkdir(parents=True, exist_ok=True)
    if train_split.resolve() != (output / 'ImageSets/train.txt').resolve():
        shutil.copyfile(train_split, output / 'ImageSets/train.txt')
    processed = output / ('waymo_processed_train_interval%d' % args.sample_interval)
    processed.mkdir(exist_ok=True)
    start = time.monotonic()
    frames = 0
    tasks = [(str(path), str(processed), args.sample_interval) for path in sources]
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                               initializer=worker_init) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, count = future.result()
            frames += count
            progress = dict(sequences_done=completed, selected_sequences=len(sources),
                            total_training_sequences=expected_sequences, frames_done=frames,
                            latest_sequence=name, elapsed_seconds=time.monotonic() - start)
            save_json(output / 'preprocess_progress.json', progress)
            print(json.dumps(progress), flush=True)
    infos = []
    for source in sources:
        path = processed / source.stem / (source.stem + '.pkl')
        contents = path.read_bytes()
        records = pickle.loads(contents)
        infos.extend(records)
    temporary = output / 'waymo_infos_train.pkl.tmp'
    with temporary.open('wb') as stream:
        pickle.dump(infos, stream)
    temporary.replace(output / 'waymo_infos_train.pkl')
    report = dict(complete=len(sources) == expected_sequences, source='Waymo training split only',
                  raw_root=str(raw_root), processed_root=str(output), raw_frame_sampling_interval=args.sample_interval,
                  loader_sampling_interval=1, sequences=len(sources), total_training_sequences=expected_sequences,
                  frames=len(infos), elapsed_seconds=time.monotonic() - start)
    save_json(output / 'preprocess_summary.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
