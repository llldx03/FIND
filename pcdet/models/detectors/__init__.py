from .second_net_iou import SECONDNetIoU

__all__ = {'SECONDNetIoU': SECONDNetIoU}


def build_detector(model_cfg, num_class, dataset):
    return __all__[model_cfg.NAME](model_cfg=model_cfg, num_class=num_class, dataset=dataset)
