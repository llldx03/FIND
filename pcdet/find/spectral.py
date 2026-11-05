"""Centered object spectra, learnable responses, and cumulative mode statistics."""
from dataclasses import dataclass
import warnings

import numpy as np
import torch
from torch import nn


def bspline_basis(x, knots, degree=3, derivative=0):
    """Clamped B-spline basis, including the right endpoint and analytic derivatives."""
    x = x.to(knots)
    right = torch.nextafter(knots[-1], knots[0])
    x = torch.where(x == knots[-1], right, x)
    if derivative:
        lower = bspline_basis(x, knots, degree - 1, derivative - 1)
        n = knots.numel() - degree - 1
        dl = knots[degree:degree + n] - knots[:n]
        dr = knots[degree + 1:degree + 1 + n] - knots[1:1 + n]
        al = torch.where(dl > 0, degree / dl.clamp_min(1e-30), 0)
        ar = torch.where(dr > 0, degree / dr.clamp_min(1e-30), 0)
        return al * lower[..., :n] - ar * lower[..., 1:n + 1]
    basis = ((x[..., None] >= knots[:-1]) & (x[..., None] < knots[1:])).to(x.dtype)
    for p in range(1, degree + 1):
        n = knots.numel() - p - 1
        dl = knots[p:p + n] - knots[:n]
        dr = knots[p + 1:p + 1 + n] - knots[1:1 + n]
        left = torch.where(dl > 0, (x[..., None] - knots[:n]) / dl.clamp_min(1e-30), 0)
        right = torch.where(dr > 0, (knots[p + 1:p + 1 + n] - x[..., None]) / dr.clamp_min(1e-30), 0)
        basis = left * basis[..., :n] + right * basis[..., 1:n + 1]
    return basis


class SplineResponse(nn.Module):
    """Cubic interpolation with h''(0)=h''(2)=0; only endpoints are parameters."""
    def __init__(self, endpoints):
        super().__init__()
        endpoints = torch.as_tensor(endpoints, dtype=torch.float64).clone()
        self.bins = endpoints.numel() - 1
        if self.bins < 2:
            raise ValueError('At least two bins are required')
        grid = torch.linspace(0, 2, self.bins + 1, dtype=torch.float64)
        knots = torch.cat((grid[:1].repeat(4), grid[1:-1], grid[-1:].repeat(4)))
        matrix = torch.cat((bspline_basis(grid, knots), bspline_basis(grid[[0, -1]], knots, derivative=2)))
        rhs = torch.cat((torch.eye(self.bins + 1, dtype=torch.float64), torch.zeros(2, self.bins + 1, dtype=torch.float64)))
        self.register_buffer('knots', knots)
        self.register_buffer('endpoint_to_control', torch.linalg.solve(matrix, rhs))
        self.endpoints = nn.Parameter(endpoints)

    def forward(self, eigenvalues):
        return (bspline_basis(eigenvalues, self.knots) @ (self.endpoint_to_control @ self.endpoints)).to(eigenvalues.dtype)

    def endpoint_basis(self, eigenvalues):
        return bspline_basis(eigenvalues, self.knots.to(eigenvalues)) @ self.endpoint_to_control.to(eigenvalues)

    @torch.no_grad()
    def project(self):
        self.endpoints.clamp_(0, 1)


class ModeMoments(nn.Module):
    """One observation per spectral mode; empty bins never add observations."""
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.register_buffer('count', torch.zeros(bins, dtype=torch.long))
        self.register_buffer('sum', torch.zeros(bins, dtype=torch.float64))
        self.register_buffer('sum_sq', torch.zeros(bins, dtype=torch.float64))

    @torch.no_grad()
    def add(self, eigenvalues, energy):
        eigenvalues = eigenvalues.detach().to(self.sum)
        energy = energy.detach().to(self.sum)
        if not torch.isfinite(energy).all() or not torch.isfinite(eigenvalues).all():
            raise FloatingPointError('Nonfinite spectral observation')
        if (energy < 0).any() or (eigenvalues < -1e-6).any() or (eigenvalues > 2 + 1e-6).any():
            raise ValueError('Invalid mode energy or normalized eigenvalue')
        index = torch.floor(eigenvalues.clamp(0, 2) * self.bins / 2).long().clamp(max=self.bins - 1)
        self.count.add_(torch.bincount(index, minlength=self.bins))
        self.sum.scatter_add_(0, index, energy)
        self.sum_sq.scatter_add_(0, index, energy.square())

    def stability(self, tau=10.0, empty=0.5):
        mean = self.sum / self.count.clamp_min(1)
        variance = (self.sum_sq / self.count.clamp_min(1) - mean.square()).clamp_min(0)
        cv = variance.sqrt() / mean.clamp_min(1e-12)
        return torch.where(self.count > 0, torch.exp(-cv / tau), torch.full_like(mean, empty))


@dataclass
class ObjectSpectrum:
    indices: torch.Tensor
    centered: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    coefficients: torch.Tensor
    endpoint_actions: torch.Tensor = None


def numpy_ilp64_eigh(laplacian):
    """Use the installed 64-bit LAPACK interface for oversized CPU workspaces."""
    from threadpoolctl import threadpool_limits

    info = np.__config__.get_info('lapack_ilp64_opt_info')
    if 'HAVE_BLAS_ILP64' not in dict(info.get('define_macros', [])):
        raise RuntimeError('Large full-graph eigendecomposition requires NumPy with ILP64 LAPACK')
    with threadpool_limits(limits=12, user_api='blas'):
        values, vectors = np.linalg.eigh(laplacian.numpy(), UPLO='L')
    return torch.from_numpy(values), torch.from_numpy(vectors)


def full_eigh(laplacian):
    nodes = laplacian.shape[0]
    workspace = 1 + 6 * nodes + 2 * nodes * nodes
    if not laplacian.is_cuda and workspace > np.iinfo(np.int32).max:
        warnings.warn(f'FIND: full {nodes}-point CPU graph uses NumPy ILP64 LAPACK')
        return numpy_ilp64_eigh(laplacian)
    return torch.linalg.eigh(laplacian)


def graph_eigh(centered, neighbors, sigma):
    distances = torch.cdist(centered, centered).square()
    distances.fill_diagonal_(float('inf'))
    values, adjacent = distances.topk(min(neighbors, len(centered) - 1), largest=False)
    del distances
    adjacency = centered.new_zeros((len(centered), len(centered)))
    adjacency.scatter_(1, adjacent, torch.exp(-values / (2 * sigma * sigma)))
    adjacency = torch.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(dim=1)
    inv = degree.clamp_min(1e-30).rsqrt()
    laplacian = adjacency
    laplacian.mul_(inv[:, None]).mul_(inv[None, :]).neg_()
    laplacian.diagonal().add_((degree > 0).to(laplacian))
    return full_eigh(laplacian)


def v2g_graph_eigh(coords, neighbors, sigma):
    """Reuse the existing voxel V2G kNN, Gaussian edges, and Laplacian."""
    from torch_cluster import knn_graph
    from pcdet.models.backbones_3d.v2g import utils

    edges = knn_graph(coords, k=min(neighbors, len(coords) - 1), loop=False)
    edges = utils._symmetrize_edges(edges)
    weights = utils._compute_edge_weights(coords, edges, sigma, 1e-6)
    laplacian = utils._normalized_laplacian(coords, edges, weights, 1e-6)
    return full_eigh(laplacian)


def use_cpu_eigh(centered):
    """Choose a device before allocating dense solver buffers."""
    if not centered.is_cuda:
        return False
    nodes = centered.shape[0]
    if nodes > 20000:
        return True
    free, _ = torch.cuda.mem_get_info(centered.device)
    cached = torch.cuda.memory_reserved(centered.device) - torch.cuda.memory_allocated(centered.device)
    available = free + max(0, cached)
    estimated_peak = 6 * nodes * nodes * centered.element_size() + 512 * 1024 ** 2
    return estimated_peak > 0.8 * available


@torch.no_grad()
def object_spectra(points, predictions, neighbors=16, sigma=0.5, score_threshold=0.2,
                   endpoint_basis=None, backend='dense', retain_vectors=True):
    """Highest-score boxes own overlaps. Unassigned/background points pass through."""
    if neighbors < 1 or sigma <= 0:
        raise ValueError('Invalid graph hyperparameters')
    if backend not in ('dense', 'v2g'):
        raise ValueError(f'Unknown graph backend: {backend}')
    decompose = v2g_graph_eigh if backend == 'v2g' else graph_eigh
    spectra = []
    for batch_index, prediction in enumerate(predictions):
        point_ids = torch.where(points[:, 0].long() == batch_index)[0]
        xyz = points[point_ids, 1:4]
        available = torch.ones(len(xyz), dtype=torch.bool, device=points.device)
        boxes = prediction['pred_boxes']
        scores = prediction['pred_scores']
        for box_id in torch.argsort(scores, descending=True):
            if scores[box_id] < score_threshold:
                continue
            box = boxes[box_id, :7]
            delta = xyz - box[:3]
            c, s = torch.cos(box[6]), torch.sin(box[6])
            local = torch.stack((delta[:, 0] * c + delta[:, 1] * s,
                                 -delta[:, 0] * s + delta[:, 1] * c, delta[:, 2]), dim=-1)
            inside = (local.abs() <= box[3:6] / 2).all(dim=-1) & available
            indices = point_ids[inside]
            if indices.numel() < 2:
                continue
            available[inside] = False
            coords = points[indices, 1:4]
            coords = coords.float() if backend == 'v2g' else coords.double()
            centered = coords - coords.mean(dim=0, keepdim=True)
            graph_coords = coords if backend == 'v2g' else centered
            cpu_fallback = use_cpu_eigh(graph_coords)
            reason = 'memory preflight'
            if not cpu_fallback:
                try:
                    eigenvalues, eigenvectors = decompose(graph_coords, neighbors, sigma)
                except torch.cuda.OutOfMemoryError:
                    if not centered.is_cuda:
                        raise
                    cpu_fallback = True
                    reason = 'CUDA allocation failure'
            if cpu_fallback:
                warnings.warn(f'FIND: full {len(indices)}-point graph eigendecomposition uses CPU ({reason})')
                threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(12)
                    eigenvalues, eigenvectors = decompose(graph_coords.cpu(), neighbors, sigma)
                finally:
                    torch.set_num_threads(threads)
            eigenvalues.clamp_(0, 2)
            coefficients = eigenvectors.T @ centered.to(eigenvectors)
            actions = None
            if endpoint_basis is not None:
                basis = endpoint_basis(eigenvalues)
                weighted = (coefficients[:, :, None] * basis[:, None, :]).reshape(len(indices), -1)
                actions = (eigenvectors @ weighted).reshape(len(indices), 3, basis.shape[1]).to(device=points.device, dtype=torch.float64)
                eigenvectors = None
            elif not retain_vectors:
                eigenvectors = None
            else:
                eigenvectors = eigenvectors.to(device=points.device, dtype=torch.float64)
            spectra.append(ObjectSpectrum(indices, centered, eigenvalues.to(points.device), eigenvectors,
                                          coefficients.to(device=points.device, dtype=torch.float64), actions))
    return spectra


class SpectralViews(nn.Module):
    def __init__(self, source_moments, tau=10.0, eta=0.2, threshold_rate=0.001):
        super().__init__()
        if (source_moments.count == 0).any():
            raise ValueError('Source prior must contain observations in every bin')
        self.tau = tau
        self.threshold_rate = threshold_rate
        self.invariant = SplineResponse(torch.cat((torch.ones(1, dtype=torch.float64, device=source_moments.sum.device), source_moments.stability(tau))))
        self.specific = SplineResponse(torch.linspace(0, 1, source_moments.bins + 1, dtype=torch.float64) ** 3)
        self.target_moments = ModeMoments(source_moments.bins)
        self.register_buffer('eta', torch.tensor(eta, dtype=torch.float64))
        nodes, weights = np.polynomial.legendre.leggauss(4)
        bins = source_moments.bins
        q = (2 * np.arange(bins)[:, None] + nodes[None, :] + 1) / bins
        self.register_buffer('quadrature_nodes', torch.from_numpy(q))
        self.register_buffer('quadrature_weights', torch.from_numpy(np.broadcast_to(weights / bins, q.shape).copy()))

    def make_views(self, points, spectra, epsilon=None, return_mask=False):
        keep = torch.ones(len(points), dtype=torch.bool, device=points.device)
        specific = points.clone()
        ratio_sum = self.eta.new_zeros(())
        ratio_count = 0
        for graph_index, graph in enumerate(spectra):
            if graph.endpoint_actions is None:
                invariant_xyz = graph.eigenvectors @ (self.invariant(graph.eigenvalues.double())[:, None] * graph.coefficients)
            else:
                invariant_xyz = graph.endpoint_actions @ self.invariant.endpoints
            norms = graph.centered.norm(dim=1)
            ratios = invariant_xyz.norm(dim=1) / norms.clamp_min(1e-12)
            selected = (ratios.detach() > self.eta) | (norms <= 1e-12)
            observed = selected & (norms > 1e-12) & torch.isfinite(ratios)
            keep[graph.indices] = selected
            ratio_sum = ratio_sum + ratios.detach()[observed].sum()
            ratio_count += int(observed.sum())
            if graph.endpoint_actions is None:
                perturbation = graph.eigenvectors @ (self.specific(graph.eigenvalues.double())[:, None] * graph.coefficients)
            else:
                perturbation = graph.endpoint_actions @ self.specific.endpoints
            coefficient = torch.rand((), device=points.device, dtype=torch.float64) * 2 - 1 if epsilon is None else epsilon
            specific[graph.indices, 1:4] = points[graph.indices, 1:4] + (coefficient * perturbation).to(points.dtype)
        result = (points[keep], specific, (ratio_sum, ratio_count))
        return result + (keep,) if return_mask else result

    @torch.no_grad()
    def accumulate(self, spectra, ratio_summary):
        for graph in spectra:
            self.target_moments.add(graph.eigenvalues, graph.coefficients.square().sum(dim=1))
        ratio_sum, ratio_count = ratio_summary
        if ratio_count:
            self.eta.lerp_(ratio_sum / ratio_count, self.threshold_rate)

    def regularizers(self):
        h_in = self.invariant(self.quadrature_nodes)
        h_sp = self.specific(self.quadrature_nodes)
        stability = self.target_moments.stability(self.tau)[:, None]
        loss_st = ((h_in * (1 - stability) + h_sp * stability) * self.quadrature_weights).sum()
        loss_comp = ((h_in + h_sp - 1).square() * self.quadrature_weights).sum()
        return loss_st, loss_comp

    @torch.no_grad()
    def project(self):
        self.invariant.project()
        self.specific.project()
