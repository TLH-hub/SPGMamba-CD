"""Differentiable superpixel (SLIC) clustering utilities.


import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_stride_and_padding(img_shape, spixel_shape):
    """
    Args:
        img_shape (Tuple[int, int]): input image shape (height, width)
        spixel_shape (Tuple[int, int]): superpixel image shape (height, width)

    Returns:
        stride (Tuple[int, int]): (stride_h, stride_w)
        padding (Tuple[int, int]): (pad_x, pad_y)
    """
    height, width = img_shape
    height_s, width_s = spixel_shape
    stride_h = (height + height_s - 1) // height_s
    stride_w = (width + width_s - 1) // width_s
    pad_y = (height_s - height % height_s) % height_s
    pad_x = (width_s - width % width_s) % width_s
    stride = (stride_h, stride_w)
    padding = (pad_x, pad_y)

    return stride, padding


def spixel_upsampling(x: torch.Tensor,
                      assignments: torch.Tensor,
                      stride: Optional[Tuple[int, int]] = None,
                      candidate_radius: int = 1) -> torch.Tensor:
    r"""upsampling a feature map based on superpixels

    Args:
        x (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
                          superpixel features
        assignments (torch.Tensor): a tensor of shape
                                    (batch, (2*candidate_radius + 1)**2, height, width)
                                    pixel-to-superpixel assignment
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_s * width_s grids
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled

    Returns:
        upsampled_features (torch.Tensor): a tensor of shape (batch, channels, height, width)
    """
    batch_size, _, height, width = assignments.shape
    n_channels = x.shape[1]
    height_s, width_s = x.shape[-2:]
    n_spixels = height_s * width_s
    if stride is None:
        stride, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    # padding an assignments so that its height and width are divisible by stride values
    pad_x, pad_y = padding
    assignments = F.pad(assignments, (0, pad_x, 0, pad_y))
    height += pad_y
    width += pad_x
    # get candidate clusters and corresponding assignments
    neighbor_range = candidate_radius * 2 + 1
    candidate_clusters = F.unfold(x, kernel_size=neighbor_range, padding=candidate_radius)
    candidate_clusters = candidate_clusters.reshape(batch_size, n_channels, neighbor_range**2, n_spixels)
    assignments = F.unfold(assignments, kernel_size=stride, stride=stride)
    assignments = assignments.reshape(batch_size, neighbor_range**2, stride[0] * stride[1], n_spixels)
    upsampled_features = torch.einsum('bkcn,bcpn->bkpn', (candidate_clusters, assignments))
    upsampled_features = upsampled_features.contiguous().reshape(batch_size * n_channels, stride[0] * stride[1], -1)
    upsampled_features = F.fold(upsampled_features, (height, width), kernel_size=stride, stride=stride)
    upsampled_features = upsampled_features.reshape(batch_size, n_channels, height, width)
    # unpad
    if pad_y > 0:
        upsampled_features = upsampled_features[..., :-pad_y, :]
    if pad_x > 0:
        upsampled_features = upsampled_features[..., :-pad_x]
    return upsampled_features


def spixel_downsampling(x: torch.Tensor,
                        assignments: torch.Tensor,
                        stride: Optional[Tuple[int, int]] = None,
                        candidate_radius: int = 1) -> torch.Tensor:
    r"""downsampling a feature map based on superpixels

    Args:
        x (torch.Tensor): a tensor of shape (batch, channels, height, width)
                          pixel features
        assignments (torch.Tensor): a tensor of shape
                                    (batch, (2*candidate_radius + 1)**2, height_s, width_s)
                                    superpixel-to-pixel assignment
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_s * width_s grids
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled

    Returns:
        downsampled_features (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
    """
    batch, _, height_s, width_s = assignments.shape
    height, width = x.shape[-2:]
    channels = x.shape[1]
    if stride is None:
        stride, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    else:
        _, padding = compute_stride_and_padding((height, width), (height_s, width_s))
    # padding an assignments so that its height and width are divisible by stride values
    pad_x, pad_y = padding
    x = F.pad(x, (0, pad_x, 0, pad_y))
    height += pad_y
    width += pad_x
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(x, kernel_size, stride=stride, padding=padding)
    unfold_elem_feats = unfold_elem_feats.reshape(batch, channels, n_candidate_pixels, height_s, width_s)
    downsampled_features = torch.einsum('bphw,bcphw->bchw', (assignments, unfold_elem_feats))
    return downsampled_features


def compute_elem_to_center_assignment(clst_feats: torch.Tensor,
                                      elem_feats: torch.Tensor,
                                      stride: Tuple[int, int],
                                      tau: float = 0.01,
                                      candidate_radius: int = 1,
                                      stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""compute elem-to-center assignment with a local attention

    Args:
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
                                   height and width should be larger than those of clst_feats
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature

    Returns:
        soft_assignment (torch.Tensor): a tensor of shape
                                        (batch, (2*candidate_radius + 1)**2, height, width)
                                        each element has a non-negative value
        similarities (torch.Tensor): a tensor of shape
                                     (batch, (2*candidate_radius + 1)**2, height, width)
                                     a similarity matrix having real values
    """
    batch_size, channels, height, width = elem_feats.shape
    n_spixels = clst_feats.shape[2] * clst_feats.shape[3]
    neighbor_range = candidate_radius * 2 + 1
    candidate_clusters = F.unfold(clst_feats, kernel_size=neighbor_range, padding=candidate_radius)
    candidate_clusters = candidate_clusters.reshape(batch_size, channels, neighbor_range**2, n_spixels)
    unfold_elem_feats = F.unfold(elem_feats, kernel_size=stride, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(batch_size, channels, stride[0] * stride[1], n_spixels)
    similarities = torch.einsum('bkcn,bkpn->bcpn', (candidate_clusters, unfold_elem_feats))
    similarities = similarities.contiguous().reshape(batch_size * neighbor_range**2, -1, n_spixels)
    similarities = F.fold(similarities, (height, width), kernel_size=stride, stride=stride)
    similarities = similarities.reshape(batch_size, neighbor_range**2, height, width)
    # masking zero padding regions with -inf
    # by using the fact that the inner product is zero.
    similarities = torch.where(similarities == 0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = (similarities / tau).softmax(1)
    return soft_assignment, similarities


def compute_center_to_elem_assignment(clst_feats: torch.Tensor,
                                      elem_feats: torch.Tensor,
                                      stride: Tuple[int, int],
                                      tau: float = 0.01,
                                      candidate_radius: int = 1,
                                      stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""compute center-to-elem assignment with a local attention

    Args:
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
                                   height and width should be larger than those of clst_feats
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature

    Returns:
        soft_assignment (torch.Tensor): a tensor of shape
                                        (batch, (2*candidate_radius + 1)**2, height_c, width_c)
                                        each element has a non-negative value
        similarities (torch.Tensor): a tensor of shape
                                     (batch, (2*candidate_radius + 1)**2, height, width)
                                     a similarity matrix having real values
    """
    b, c, h, w = clst_feats.shape
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0] * neighbor_range, stride[1] * neighbor_range)
    padding = (stride[0] * candidate_radius, stride[1] * candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(elem_feats, kernel_size, padding=padding, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w)
    similarities = torch.einsum('bcphw,bchw->bphw', (unfold_elem_feats, clst_feats))
    similarities = torch.where(similarities == 0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = torch.softmax(similarities / tau, dim=1)
    return soft_assignment, similarities


def update_clst_feats(elem_feats: torch.Tensor,
                      clst_feats: torch.Tensor,
                      stride: Tuple[int, int],
                      tau: float = 0.01,
                      candidate_radius: int = 1,
                      stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""update cluster features with a local attention. this function equivalent to
    compute_center_to_elem_assignment with spixel_downsampling

    Args:
        elem_feats (torch.Tensor): a tensor of shape (batch, channels, height, width)
                                   height and width should be larger than those of clst_feats
        clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        stride (Tuple[int, int]): grid size when dividing elem_feats into height_c * width_c grids
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature

    Returns:
        new_clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_c, width_c)
        soft_assignment (torch.Tensor): a tensor of shape
                                        (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_c, width_c)
                                        each element has a non-negative value
        similarities (torch.Tensor): a tensor of shape
                                     (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_c, width_c)
                                     a similarity matrix having real values
    """
    b, c, h, w = clst_feats.shape
    neighbor_range = candidate_radius * 2 + 1
    kernel_size = (stride[0]*neighbor_range, stride[1]*neighbor_range)
    padding = (stride[0]*candidate_radius, stride[1]*candidate_radius)
    n_candidate_pixels = kernel_size[0] * kernel_size[1]
    unfold_elem_feats = F.unfold(elem_feats, kernel_size, padding=padding, stride=stride)
    unfold_elem_feats = unfold_elem_feats.reshape(b, c, n_candidate_pixels, h, w)
    similarities = torch.einsum('bcphw,bchw->bphw', (unfold_elem_feats, clst_feats))
    similarities = torch.where(similarities == 0, -torch.inf, similarities)
    if stable:
        similarities = similarities - similarities.max(1, keepdim=True).values.detach()
    soft_assignment = torch.softmax(similarities / tau, dim=1)
    new_clst_feats = torch.einsum('bphw,bcphw->bchw', (soft_assignment, unfold_elem_feats))
    return new_clst_feats, soft_assignment, similarities


class DiffSLIC(nn.Module):
    r"""Differentiable SLIC

    Args:
        n_spixels (int): a number of superpixels
        n_iter (int): a number of iterations for updating cluster centers
        tau (float): a temperature parameter. when tau -> 0, assignment is deterministic
        normalize (bool): if True, pixel and superpixel features are normalized so that their l2 norm is 1
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled
        stable (bool): if True, using stable computation of softmax with temperature
                       `stable` should be True, when using extremely small tau for obtaining deterministic assignment.
    """
    def __init__(self,
                 n_spixels: int,
                 n_iter: int = 5,
                 tau: float = 0.01,
                 candidate_radius: int = 2,
                 normalize: bool = True,
                 stable: bool = False) -> None:
        super().__init__()
        self.n_spixels = n_spixels
        self.n_iter = n_iter
        self.tau = tau
        self.candidate_radius = candidate_radius
        self.normalize = normalize
        self.stable = stable

    def forward(self, x: torch.Tensor,
                clst_feats: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
            x (torch.Tensor): a tensor of shape (batch, channels, height, width)
            clst_feats (Optional[torch.Tensor]): a tensor of shape (batch, channels, height_s, width_s)
                                                 initial cluster features. if clst_feats is None, it is
                                                 initialized by averaging pixels in a uniform grid

        Returns:
            clst_feats (torch.Tensor): a tensor of shape (batch, channels, height_s, width_s)
                                       height_s * width_s <= self.n_spixels
            p2s_assign (torch.Tensor): a tensor of shape (batch, (2*candidate_radius + 1)**2, height, width)
                                       a pixel-to-superpixel assignment matrix
            s2p_assign (torch.Tensor): a tensor of shape
                                       (batch, stride_h * stride_w * (2*candidate_radius + 1)**2, height_s, width_s)
                                       a superpixel-to-pixel assignment matrix
                                       if n_iter is 0, s2p_assign is None
        """
        height, width = x.shape[-2:]
        # initialize cluster features
        if clst_feats is None:
            height_s = int(math.sqrt(self.n_spixels * height / width))
            width_s = int(math.sqrt(self.n_spixels * width / height))
            stride_h = (height + height_s - 1) // height_s
            stride_w = (width + width_s - 1) // width_s
            stride = (stride_h, stride_w)
            clst_feats = F.adaptive_avg_pool2d(x, (height_s, width_s))
        else:
            height_s, width_s = clst_feats.shape[-2:]
            stride = ((height + height_s) // height_s, (width + width_s) // width_s)
        # normalize feature vectors so that their l2-norm is 1
        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True)
            clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True)
        # padding an image feature so that its height and width are divisible by stride values
        pad_x = (width_s - width % width_s) % width_s
        pad_y = (height_s - height % height_s) % height_s
        x = F.pad(x, (0, pad_x, 0, pad_y))
        # update cluster features
        s2p_assign = None
        for _ in range(self.n_iter):
            clst_feats, s2p_assign, _ = update_clst_feats(x, clst_feats, stride, self.tau, self.candidate_radius)
            if self.normalize:
                clst_feats = clst_feats / clst_feats.norm(dim=1, keepdim=True)
        # compute a pixel-to-superpixel assignment
        p2s_assign, _ = compute_elem_to_center_assignment(clst_feats, x, stride, self.tau, self.candidate_radius)
        # remove the padding region
        if pad_y > 0:
            p2s_assign = p2s_assign[..., :-pad_y, :]
        if pad_x > 0:
            p2s_assign = p2s_assign[..., :-pad_x]
        return clst_feats, p2s_assign, s2p_assign

    def extra_repr(self):
        return f'n_spixels={self.n_spixels}, \n ' \
               f'n_iter={self.n_iter}, \n ' \
               f'tau={self.tau}, \n ' \
               f'candidate_radius={self.candidate_radius}, \n'


def lbl_to_rgb(lbl: np.ndarray,
               color_palette: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """integer label to (random) rgb image

    Args:
        lbl (np.ndarray): an array of shape (*, height, width)
                          each element denotes an integer label
        color_palette (np.ndarray): an array of shape (n_segments, 3)
                                    each row denotes rgb

    Returns:
        rgb_lbl (np.ndarray): an array of shape (*, height, width, 3)
                              each element is assigned a rgb value based on color_palette
        color_palette (np.ndarray): an array of shape (n_segments, 3)
                                    if color_palette is not given as input,
                                    n_segments is height * width
    """
    height, width = lbl.shape[-2:]
    if color_palette is None:
        n_segments = height * width
        color_palette = np.random.randint(0, 255, (n_segments, 3), dtype=np.uint8)
    return color_palette[lbl], color_palette


class ConnDiffSLIC(DiffSLIC):
    r"""DiffSLIC with additional connectivity enforcement.

    Applies a median-filtering refinement to the pixel-to-superpixel
    assignment so that the resulting superpixels remain spatially connected.

    Args:
        n_spixels (int): a number of superpixels.
        n_iter (int): a number of iterations for updating cluster centers.
        tau (float): a temperature parameter.
        candidate_radius (int): a radius of the region from which the candidate clusters are sampled.
        normalize (bool): if True, pixel and superpixel features are normalized so that their l2 norm is 1.
        stable (bool): if True, using stable computation of softmax with temperature.
        conn_iterations (int): a number of connectivity refinement iterations.
    """

    def __init__(
        self,
        n_spixels: int,
        n_iter: int = 5,
        tau: float = 0.01,
        candidate_radius: int = 2,
        normalize: bool = True,
        stable: bool = False,
        conn_iterations: int = 3,
    ) -> None:
        super().__init__(n_spixels, n_iter, tau, candidate_radius, normalize, stable)
        self.conn_iterations = conn_iterations

    @torch.no_grad()
    def enforce_connectivity_gpu(self, assign: torch.Tensor, h_s: int, w_s: int) -> torch.Tensor:
        r"""Enforce spatial connectivity on a pixel-to-superpixel assignment.

        Args:
            assign (torch.Tensor): pixel-to-superpixel assignment of shape
                                   (batch, (2*candidate_radius + 1)**2, height, width).
            h_s (int): superpixel grid height.
            w_s (int): superpixel grid width.

        Returns:
            torch.Tensor: hard superpixel label map of shape (batch, 1, height, width).
        """
        B, _, H, W = assign.shape
        device = assign.device

        stride_h = (H + h_s - 1) // h_s
        stride_w = (W + w_s - 1) // w_s

        y_coords = torch.arange(H, device=device) // stride_h
        x_coords = torch.arange(W, device=device) // stride_w
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        grid_y = torch.clamp(grid_y, 0, h_s - 1)
        grid_x = torch.clamp(grid_x, 0, w_s - 1)

        r = self.candidate_radius
        side = 2 * r + 1
        off_idx = assign.argmax(1)  # (B, H, W)
        off_y = (off_idx // side) - r
        off_x = (off_idx % side) - r

        res_y = torch.clamp(grid_y.unsqueeze(0) + off_y, 0, h_s - 1)
        res_x = torch.clamp(grid_x.unsqueeze(0) + off_x, 0, w_s - 1)
        label_map = (res_y * w_s + res_x).unsqueeze(1).float()

        refined_label = label_map
        for _ in range(self.conn_iterations):
            padded = F.pad(refined_label, (1, 1, 1, 1), mode='replicate')
            unfolded = F.unfold(padded, kernel_size=3)
            refined_label, _ = torch.median(unfolded, dim=1, keepdim=True)
            refined_label = refined_label.view(B, 1, H, W)

        return refined_label.long()

    def forward(
        self,
        x: torch.Tensor,
        clst_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Run DiffSLIC and attach a connectivity-refined label map.

        Args:
            x (torch.Tensor): input image features of shape (batch, channels, height, width).
            clst_feats (Optional[torch.Tensor]): initial cluster features.

        Returns:
            tuple: clst_feats, p2s_assign, s2p_assign, and a connectivity-refined label map.
        """
        clst_feats, p2s_assign, s2p_assign = super().forward(x, clst_feats)
        h_s, w_s = clst_feats.shape[-2:]
        conn_label_hard = self.enforce_connectivity_gpu(p2s_assign, h_s, w_s).float()
        p2s_max_prob = p2s_assign.max(dim=1, keepdim=True)[0]
        conn_label = (conn_label_hard - p2s_max_prob).detach() + p2s_max_prob
        return clst_feats, p2s_assign, s2p_assign, conn_label
