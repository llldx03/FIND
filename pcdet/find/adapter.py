"""Predict-before-update FIND with independent student and EMA teacher detectors."""
import random

import numpy as np
import torch

from .spectral import object_spectra
from .voxelize import graph_points, voxel_views


def release_cuda_workspace(points):
    """Return unused cached allocations before external CUDA workspace requests."""
    if points.is_cuda:
        torch.cuda.empty_cache()


def input_only(batch):
    """Restrict the detector input to points, voxels, and evaluation metadata."""
    allowed = ('points', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size',
               'frame_id', 'calib', 'image_shape', 'metadata', 'use_lead_xyz')
    return {key: batch[key] for key in allowed if key in batch}


def empty_prediction(points):
    return dict(pred_boxes=points.new_empty((0, 7)), pred_scores=points.new_empty(0),
                pred_labels=torch.empty(0, dtype=torch.long, device=points.device))


@torch.no_grad()
def detector_predictions(model, batch):
    if not len(batch['voxels']):
        return [empty_prediction(batch['points']) for _ in range(batch['batch_size'])]
    predictions, _ = model(batch)
    occupied = set(batch['voxel_coords'][:, 0].long().unique().tolist())
    return [prediction if index in occupied else empty_prediction(batch['points'])
            for index, prediction in enumerate(predictions)]


def pseudo_boxes(predictions, positive_thresholds, negative_thresholds):
    reference = predictions[0]['pred_boxes']
    batches = []
    for prediction in predictions:
        labels = prediction['pred_labels'].long()
        scores = prediction['pred_scores']
        positive = scores.new_tensor(positive_thresholds)[labels - 1]
        negative = scores.new_tensor(negative_thresholds)[labels - 1]
        keep = scores >= negative
        labels = torch.where(scores >= positive, labels, -labels)
        batches.append(torch.cat((prediction['pred_boxes'][keep, :7], labels[keep, None]), dim=1).detach())
    padded = reference.new_zeros((len(batches), max(1, max(len(x) for x in batches)), 8))
    for index, boxes in enumerate(batches):
        padded[index, :len(boxes)] = boxes
    return padded


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


class FINDAdapter:
    def __init__(self, student, teacher, views, voxelizer, detector_optimizer,
                 filter_optimizer, scheduler, config, positive_thresholds, negative_thresholds):
        self.student, self.teacher = student, teacher
        self.views, self.voxelizer = views, voxelizer
        self.detector_optimizer, self.filter_optimizer = detector_optimizer, filter_optimizer
        self.scheduler, self.config = scheduler, config
        self.positive_thresholds, self.negative_thresholds = positive_thresholds, negative_thresholds
        self.graph_representation = getattr(config, 'GRAPH_REPRESENTATION', 'points')
        self.step_index = 0
        self.samples_seen = 0
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def graph_arguments(self):
        return dict(neighbors=self.config.NUM_NEIGHBORS, sigma=self.config.RBF_SIGMA,
                    score_threshold=self.config.GRAPH_SCORE_THRESHOLD,
                    backend=getattr(self.config, 'GRAPH_BACKEND', 'dense'))

    def predict_views(self, batch):
        base = input_only(batch)
        release_cuda_workspace(base['points'])
        self.teacher.eval()
        with torch.no_grad():
            reference = detector_predictions(self.teacher, dict(base))
            nodes = graph_points(base, self.graph_representation)
            spectra = object_spectra(nodes, reference, endpoint_basis=self.views.invariant.endpoint_basis,
                                     **self.graph_arguments())
        invariant_points, specific_points, ratio_summary, keep = self.views.make_views(nodes, spectra, return_mask=True)
        if self.graph_representation == 'voxel':
            invariant_batch, specific_batch = voxel_views(base, specific_points, keep)
        else:
            invariant_batch = self.voxelizer(invariant_points, base['batch_size'])
        release_cuda_workspace(base['points'])
        with torch.no_grad():
            predictions = detector_predictions(self.teacher, invariant_batch)
            predictions = [{k: v.detach().clone() for k, v in p.items() if k in ('pred_boxes', 'pred_scores', 'pred_labels')}
                           for p in predictions]
        if self.graph_representation != 'voxel':
            specific_batch = self.voxelizer(specific_points, base['batch_size'])
        return predictions, specific_batch, spectra, ratio_summary, len(invariant_points)

    @torch.no_grad()
    def update_teacher(self):
        student = dict(self.student.named_parameters())
        for name, parameter in self.teacher.named_parameters():
            parameter.lerp_(student[name], 1 - self.config.DETECTOR_EMA)
        buffers = dict(self.student.named_buffers())
        for name, buffer in self.teacher.named_buffers():
            buffer.copy_(buffers[name])

    def step(self, batch):
        predictions, specific_batch, spectra, ratio_summary, retained = self.predict_views(batch)
        specific_batch['gt_boxes'] = pseudo_boxes(predictions, self.positive_thresholds, self.negative_thresholds)
        self.views.accumulate(spectra, ratio_summary)
        self.scheduler.step(self.step_index)
        learning_rate = float(self.detector_optimizer.lr)
        momentum = float(self.detector_optimizer.mom)
        for group in self.filter_optimizer.param_groups:
            group['lr'] = learning_rate
            group['betas'] = (momentum, 0.99)
        self.detector_optimizer.zero_grad()
        self.filter_optimizer.zero_grad(set_to_none=True)
        self.student.train()
        has_specific_voxels = bool(len(specific_batch['voxels']))
        if has_specific_voxels:
            release_cuda_workspace(batch['points'])
            result, _, _ = self.student(specific_batch)
            loss_cons = result['loss'].mean()
        else:
            loss_cons = self.views.specific.endpoints.sum() * 0
        loss_st, loss_comp = self.views.regularizers()
        loss = loss_cons + self.config.BETA_ST * loss_st + self.config.BETA_COMP * loss_comp
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite FIND objective')
        release_cuda_workspace(batch['points'])
        loss.backward()
        parameters = [p for p in list(self.student.parameters()) + list(self.views.parameters()) if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.GRAD_NORM_CLIP, error_if_nonfinite=True)
        invariant_grad = float(self.views.invariant.endpoints.grad.norm())
        specific_grad = float(self.views.specific.endpoints.grad.norm())
        if has_specific_voxels:
            self.detector_optimizer.step()
        self.filter_optimizer.step()
        self.views.project()
        if has_specific_voxels:
            self.student.update_global_step()
            self.update_teacher()
        self.step_index += 1
        self.samples_seen += int(batch['batch_size'])
        metrics = dict(step=self.step_index, samples_seen=self.samples_seen, loss=float(loss.detach()),
                       loss_cons=float(loss_cons.detach()), loss_st=float(loss_st.detach()),
                       loss_comp=float(loss_comp.detach()), learning_rate=learning_rate,
                       grad_norm=float(grad_norm), invariant_grad=invariant_grad, specific_grad=specific_grad,
                       eta=float(self.views.eta),
                       graph_count=len(spectra), graph_modes=sum(len(g.indices) for g in spectra),
                       largest_graph=max((len(g.indices) for g in spectra), default=0),
                       graph_representation=self.graph_representation,
                       input_points=len(batch['points']), input_voxels=len(batch['voxels']),
                       invariant_points=retained if self.graph_representation == 'points' else None,
                       invariant_voxels=retained if self.graph_representation == 'voxel' else None,
                       positive_pseudo_labels=int((specific_batch['gt_boxes'][..., -1] > 0).sum()),
                       mode_counts=self.views.target_moments.count.tolist())
        return predictions, metrics

    def state_dict(self):
        return dict(format_version=1, student=self.student.state_dict(), teacher=self.teacher.state_dict(),
                    views=self.views.state_dict(), detector_optimizer=self.detector_optimizer.state_dict(),
                    filter_optimizer=self.filter_optimizer.state_dict(),
                    scheduler_step=self.step_index, step_index=self.step_index, samples_seen=self.samples_seen,
                    rng=capture_rng())

    def load_state_dict(self, state):
        if state['format_version'] != 1:
            raise ValueError('Unsupported FIND checkpoint format')
        step_index = int(state['step_index'])
        samples_seen = int(state['samples_seen'])
        if state['scheduler_step'] != step_index or step_index < 0 or samples_seen < 0:
            raise ValueError('Invalid checkpoint progress')
        self.student.load_state_dict(state['student'], strict=True)
        self.teacher.load_state_dict(state['teacher'], strict=True)
        self.views.load_state_dict(state['views'], strict=True)
        self.detector_optimizer.load_state_dict(state['detector_optimizer'])
        self.filter_optimizer.load_state_dict(state['filter_optimizer'])
        self.step_index = step_index
        self.samples_seen = samples_seen
        restore_rng(state['rng'])
