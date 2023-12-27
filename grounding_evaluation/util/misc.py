# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
import torchvision
from torch import Tensor
import math


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def video_collate_fn(do_round, div_vid, batch):
    # div_vid: if set >0, each video is divided into clips of div_vid (number of frames that the model takes as input) consecutive frames
    batch = list(zip(*batch))
    final_batch = {}
    final_batch["samples"] = NestedTensor.from_tensor_list(
        batch[0], do_round
    )  # padded on the longest height and width
    if len(batch) == 4:  # stride > 0
        final_batch["samples_fast"] = NestedTensor.from_tensor_list(
            batch[3], do_round
        )  # padded on the longest height and width
    final_batch["durations"] = [
        len(x) for x in batch[1]
    ]  # used to temporally unpad samples later
    final_batch["targets"] = [
        target for clip in batch[1] for target in clip
    ]  # flatten list of list of targets (one list per video), not temporally padded
    final_batch["captions"] = [
        tmp_target["caption"] for tmp_target in batch[2]
    ]  # one caption per clip
    final_batch["video_ids"] = [tmp_target["video_id"] for tmp_target in batch[2]]
    final_batch["frames_id"] = [tmp_target["frames_id"] for tmp_target in batch[2]]
    final_batch["inter_idx"] = [tmp_target["inter_idx"] for tmp_target in batch[2]]
    if "qtype" in batch[2][0]:
        final_batch["qtype"] = [tmp_target["qtype"] for tmp_target in batch[2]]
        final_batch["qtype"] = {
            video_id: qtype
            for video_id, qtype in zip(final_batch["video_ids"], final_batch["qtype"])
        }

    if div_vid:
        n_fwds = [
            math.ceil(t / div_vid) for t in final_batch["durations"]
        ]  # number of forwards for each video
        final_batch["durations"] = [
            min(div_vid, t - i_clip * div_vid)
            for i_dur, t in enumerate(final_batch["durations"])
            for i_clip in range(n_fwds[i_dur])
        ]  # update duration for each clip
        final_batch["captions"] = [
            caption
            for i_cap, caption in enumerate(final_batch["captions"])
            for _ in range(n_fwds[i_cap])
        ]  # repeat captions, one per forward
        final_batch["video_ids"] = [
            video_id
            for i_vid, video_id in enumerate(final_batch["video_ids"])
            for _ in range(n_fwds[i_vid])
        ]  # repeat video ids, one per forward
        final_inter_idx = []  # update annotated time interval for each of the clip
        for i_b, inter_idx in enumerate(final_batch["inter_idx"]):
            for i_clip in range(n_fwds[i_b]):
                start, end = inter_idx
                max_start = max(i_clip * div_vid, start)
                min_end = min((i_clip + 1) * div_vid - 1, end)
                if max_start > min_end:
                    final_inter_idx.append([-100, -100])
                else:
                    final_inter_idx.append(
                        [max_start - i_clip * div_vid, min_end - i_clip * div_vid]
                    )
        final_batch["inter_idx"] = final_inter_idx

    return final_batch


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        # TODO make this more general
        if tensor_list[0].ndim == 3:  # images
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        elif tensor_list[0].ndim == 4:  # videos
            max_size = tuple(max(s) for s in zip(*[clip.shape for clip in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, t, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, t, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device

            nb_images = sum(
                clip.shape[1] for clip in tensor_list
            )  # total number of frames in the batch
            tensor = torch.zeros((nb_images, c, h, w), dtype=dtype, device=device)
            mask = torch.ones((nb_images, h, w), dtype=torch.bool, device=device)
            cur_dur = 0
            for i_clip, clip in enumerate(tensor_list):
                tensor[
                    cur_dur : cur_dur + clip.shape[1],
                    : clip.shape[0],
                    : clip.shape[2],
                    : clip.shape[3],
                ].copy_(clip.transpose(0, 1))
                mask[
                    cur_dur : cur_dur + clip.shape[1], : clip.shape[2], : clip.shape[3]
                ] = False
                cur_dur += clip.shape[1]
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "image_id",
    ]
    return [
        {k: v.to(device) if k not in excluded_keys else v for k, v in t.items()}
        for t in targets
    ]


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)