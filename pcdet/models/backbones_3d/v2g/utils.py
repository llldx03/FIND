"""Voxel graph construction shared by FIND object spectra."""
import torch


def _symmetrize_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Ensure the kNN graph is treated as undirected."""
    reversed_edges = edge_index[[1, 0]]
    edge_index = torch.cat([edge_index, reversed_edges], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def _compute_edge_weights(points: torch.Tensor, edge_index: torch.Tensor, sigma: float, eps: float) -> torch.Tensor:
    row, col = edge_index
    diff = points[row] - points[col]
    dist_sq = (diff * diff).sum(dim=1).clamp_min(eps)
    if sigma > 0:
        coeff = -dist_sq / (2.0 * (sigma ** 2))
        edge_weight = torch.exp(coeff)
    else:
        edge_weight = 1.0 / torch.sqrt(dist_sq)
    return edge_weight


def _normalized_laplacian(points, edge_index, edge_weight, eps):
    num_nodes = points.size(0)
    row, col = edge_index
    adj = points.new_zeros((num_nodes, num_nodes))
    adj.index_put_((row, col), edge_weight.to(points.dtype), accumulate=True)
    degree = adj.sum(dim=1).clamp_min(eps)
    deg_inv_sqrt = degree.pow(-0.5)
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return torch.eye(num_nodes, device=points.device, dtype=points.dtype) - norm_adj
