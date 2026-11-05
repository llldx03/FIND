"""Train the source SECOND-IoU detector on one GPU or with torchrun."""
import argparse
import os
from pathlib import Path

import _init_path
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.find.runtime import set_seed
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size per GPU; defaults to the config')
    parser.add_argument('--workers', type=int, default=4, help='Data workers per GPU')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', help='Source training checkpoint, including optimizer state')
    parser.add_argument('--pretrained', help='Initialize model weights without resuming the optimizer')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--keep-checkpoints', type=int, default=5)
    parser.add_argument('--set', dest='overrides', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.resume and args.pretrained:
        parser.error('Choose either --resume or --pretrained')
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(args.overrides, cfg)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = world > 1
    torch.cuda.set_device(local_rank)
    cfg.LOCAL_RANK = local_rank
    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
    set_seed(args.seed + rank)
    batch_size = args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    epochs = args.epochs or cfg.OPTIMIZATION.NUM_EPOCHS
    if min(batch_size, epochs, args.save_every, args.keep_checkpoints) < 1 or args.workers < 0:
        parser.error('Invalid batch size, epoch count, checkpoint interval, or worker count')
    output = Path(args.output)
    checkpoint_dir = output / 'ckpt'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(str(output / 'train.log'), rank=rank)
    logger.info('Source training: world_size=%d, batch_size_per_gpu=%d, epochs=%d', world, batch_size, epochs)
    log_config_to_file(cfg, logger=logger)
    dataset, loader, sampler = build_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size, distributed,
        workers=args.workers, logger=logger, training=True,
    )
    if len(loader) == 0:
        raise ValueError('Source dataset is empty')
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    start_iter, start_epoch = 0, 0
    if args.pretrained:
        model.load_params_from_file(args.pretrained, logger, to_cpu=True)
    if args.resume:
        start_iter, start_epoch = model.load_params_with_optimizer(
            args.resume, optimizer=optimizer, logger=logger,
        )
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True,
        )
    scheduler, warmup = build_scheduler(optimizer, len(loader), epochs, start_epoch - 1, cfg.OPTIMIZATION)
    writer = SummaryWriter(str(output / 'tensorboard')) if rank == 0 else None
    train_model(
        model, optimizer, loader, model_fn_decorator(), scheduler, cfg.OPTIMIZATION,
        start_epoch, epochs, int(start_iter), rank, writer, checkpoint_dir,
        source_sampler=sampler, lr_warmup_scheduler=warmup,
        ckpt_save_interval=args.save_every, max_ckpt_save_num=args.keep_checkpoints,
    )
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
