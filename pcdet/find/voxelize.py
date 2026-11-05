"""Hard voxel assignment with differentiable point features."""
import torch


def graph_points(batch, representation='points'):
    """Expose the existing MeanVFE voxel means with their batch indices."""
    if representation == 'points':
        return batch['points']
    if representation != 'voxel':
        raise ValueError(f'Unknown graph representation: {representation}')
    voxels = batch['voxels']
    normalizer = batch['voxel_num_points'].view(-1, 1).clamp_min(1.0).type_as(voxels)
    features = voxels.sum(dim=1) / normalizer
    return torch.cat((batch['voxel_coords'][:, :1].to(features), features), dim=1)


def voxel_views(batch, specific_points, keep):
    """Keep V2G's voxel coordinates; feed changed means through MeanVFE."""
    invariant = dict(batch)
    for name in ('voxels', 'voxel_coords', 'voxel_num_points'):
        invariant[name] = batch[name][keep]
    specific = dict(batch)
    specific['voxels'] = specific_points[:, None, 1:].contiguous()
    specific['voxel_num_points'] = torch.ones_like(batch['voxel_num_points'])
    return invariant, specific


class DifferentiableVoxelizer:
    def __init__(self, point_cloud_range, voxel_size, max_points, max_voxels):
        self.point_cloud_range = list(point_cloud_range)
        self.voxel_size = list(voxel_size)
        self.max_points = int(max_points)
        self.max_voxels = int(max_voxels)

    def __call__(self, points, batch_size):
        device = points.device
        limits = points.new_tensor(self.point_cloud_range)
        size = points.new_tensor(self.voxel_size)
        grid = torch.round((limits[3:] - limits[:3]) / size).long()
        batch_voxels, batch_coords, batch_counts = [], [], []
        for batch_index in range(batch_size):
            values = points[points[:, 0].long() == batch_index, 1:]
            coords = torch.floor((values[:, :3].detach() - limits[:3]) / size).long()
            valid = ((coords >= 0) & (coords < grid)).all(dim=1)
            values, coords = values[valid], coords[valid]
            if not len(values):
                continue
            ids = (coords[:, 2] * grid[1] + coords[:, 1]) * grid[0] + coords[:, 0]
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            order = torch.argsort(inverse, stable=True)
            sorted_inverse = inverse[order]
            start = torch.cat((torch.zeros(1, dtype=torch.long, device=device),
                               torch.where(sorted_inverse[1:] != sorted_inverse[:-1])[0] + 1))
            first = order[start]
            voxel_order = torch.argsort(first)[:self.max_voxels]
            remap = torch.full((len(unique),), -1, dtype=torch.long, device=device)
            remap[voxel_order] = torch.arange(len(voxel_order), device=device)
            voxel_ids = remap[inverse]
            ranks_sorted = torch.arange(len(order), device=device) - start[sorted_inverse]
            ranks = torch.empty_like(ranks_sorted)
            ranks[order] = ranks_sorted
            keep = (voxel_ids >= 0) & (ranks < self.max_points)
            voxels = values.new_zeros((len(voxel_order), self.max_points, values.shape[1]))
            voxels[voxel_ids[keep], ranks[keep]] = values[keep]
            counts = torch.bincount(voxel_ids[keep], minlength=len(voxel_order)).int()
            xyz = coords[first[voxel_order]]
            b = torch.full((len(xyz), 1), batch_index, device=device, dtype=torch.long)
            batch_voxels.append(voxels)
            batch_coords.append(torch.cat((b, xyz[:, [2, 1, 0]]), dim=1).int())
            batch_counts.append(counts)
        if not batch_voxels:
            return dict(points=points, voxels=points.new_empty((0, self.max_points, points.shape[1] - 1)),
                        voxel_coords=torch.empty((0, 4), dtype=torch.int32, device=device),
                        voxel_num_points=torch.empty(0, dtype=torch.int32, device=device), batch_size=batch_size)
        return dict(points=points, voxels=torch.cat(batch_voxels),
                    voxel_coords=torch.cat(batch_coords), voxel_num_points=torch.cat(batch_counts),
                    batch_size=batch_size)

    @classmethod
    def from_dataset(cls, dataset):
        configs = [c for c in dataset.dataset_cfg.DATA_PROCESSOR if c.NAME == 'transform_points_to_voxels']
        encoding = dataset.point_feature_encoder.point_encoding_config
        if len(configs) != 1 or encoding.encoding_type != 'absolute_coordinates_encoding':
            raise ValueError('FIND requires one voxelizer with XYZ point features')
        config = configs[0]
        return cls(dataset.point_cloud_range, config.VOXEL_SIZE, config.MAX_POINTS_PER_VOXEL,
                   config.MAX_NUMBER_OF_VOXELS['test'])
