# Adapted from https://github.com/hassony2/torch_videovision
from PIL import Image
import numbers
import torch
import cv2
import numpy as np
import PIL
from PIL import Image


def convert_img(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format"""
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor(object):
    """Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, "Got {0} instead of 3 channels".format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(
                "Expected numpy.ndarray or PIL.Image\
            but got list of {0}".format(
                    type(clip[0])
                )
            )

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(
                    "Expected numpy.ndarray or PIL.Image\
                but got list of {0}".format(
                        type(clip[0])
                    )
                )
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip

        else:
            tensor_clip = torch.from_numpy(np_clip)

            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h : min_h + h, min_w : min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
    return cropped


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError("tensor is not a torch clip.")

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def resize_clip(clip, size, interpolation="bilinear"):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == "bilinear":
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError(
            "Expected numpy.ndarray or PIL.Image"
            + "but got list of {0}".format(type(clip[0]))
        )
    return scaled



# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from datasets.torch_videovision import ClipToTensor, normalize, resize_clip, crop_clip
from util.box_ops import box_xyxy_to_cxcywh
import torch
import random
import numpy as np
import copy
import PIL
from util.misc import interpolate


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video, targets):
        for t in self.transforms:
            video, targets = t(video, targets)
        return video, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.ClipToTensor = ClipToTensor(channel_nb, div_255, numpy)

    def __call__(self, video, targets):
        return self.ClipToTensor(video), targets


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, video, targets):
        video = normalize(
            video, mean=self.mean, std=self.std
        )  # torch functional videotransforms
        if targets is None:
            return video, None
        targets = targets.copy()
        h, w = video.shape[-2:]
        if "boxes" in targets[0]:  # apply for every image of the clip
            for i_tgt in range(len(targets)):
                boxes = targets[i_tgt]["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                targets[i_tgt]["boxes"] = boxes
        return video, targets


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video, targets):
        if random.random() < self.p:
            return hflip(video, targets)
        return video, targets


def hflip(clip, targets):
    if isinstance(clip[0], np.ndarray):
        flipped_clip = [
            np.fliplr(img) for img in clip
        ]  # apply for every image of the clip
    elif isinstance(clip[0], PIL.Image.Image):
        flipped_clip = [
            img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
        ]  # apply for every image of the clip

    w, h = clip[0].size

    targets = targets.copy()
    if "boxes" in targets[0]:  # apply for every image of the clip
        for i_tgt in range(len(targets)):
            boxes = targets[i_tgt]["boxes"]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                [-1, 1, -1, 1]
            ) + torch.as_tensor([w, 0, w, 0])
            targets[i_tgt]["boxes"] = boxes

    if "masks" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = targets[i_tgt]["masks"].flip(-1)

    if (
        "caption" in targets[0]
    ):  # TODO: quick hack, only modify the first one as all of them should be the same
        caption = (
            targets[0]["caption"]
            .replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        targets[0]["caption"] = caption

    return flipped_clip, targets


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, video, targets):
        if random.random() < self.p:
            return self.transforms1(video, targets)
        return self.transforms2(video, targets)


def resize(clip, targets, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if isinstance(clip[0], PIL.Image.Image):
        s = clip[0].size
    elif isinstance(clip[0], np.ndarray):
        h, w, ch = list(clip[0].shape)
        s = [w, h]
    else:
        raise NotImplementedError
    size = get_size(
        s, size, max_size
    )  # apply for first image, all images of the same clip have the same h w
    rescaled_clip = resize_clip(clip, size)  # torch video transforms functional
    if isinstance(clip[0], np.ndarray):
        h2, w2, c2 = list(rescaled_clip[0].shape)
        s2 = [w2, h2]
    elif isinstance(clip[0], PIL.Image.Image):
        s2 = rescaled_clip[0].size
    else:
        raise NotImplementedError

    if targets is None:
        return rescaled_clip, None

    ratios = tuple(float(s_mod) / float(s_orig) for s_mod, s_orig in zip(s2, s))
    ratio_width, ratio_height = ratios

    targets = targets.copy()
    if "boxes" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            boxes = targets[i_tgt]["boxes"]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height]
            )
            targets[i_tgt]["boxes"] = scaled_boxes

    if (
        "area" in targets[0]
    ):  # TODO: not sure if it is needed to do for all images from the clip
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            area = targets[i_tgt]["area"]
            scaled_area = area * (ratio_width * ratio_height)
            targets[i_tgt]["area"] = scaled_area

    h, w = size
    for i_tgt in range(
        len(targets)
    ):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    if "masks" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = (
                interpolate(
                    targets[i_tgt]["masks"][:, None].float(), size, mode="nearest"
                )[:, 0]
                > 0.5
            )

    return rescaled_clip, targets


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, video, target=None):
        size = random.choice(self.sizes)
        return resize(video, target, size, self.max_size)


def crop(clip, targets, region):
    cropped_clip = crop_clip(clip, *region)
    # cropped_clip = [F.crop(img, *region) for img in clip] # other possibility is to use torch_videovision.torchvideotransforms.functional.crop_clip

    targets = targets.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    for i_tgt in range(
        len(targets)
    ):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map", "isfinal"]

    if "boxes" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            boxes = targets[i_tgt]["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            targets[i_tgt]["boxes"] = cropped_boxes.reshape(-1, 4)
            targets[i_tgt]["area"] = area
        fields.append("boxes")

    if "masks" in targets[0]:
        # FIXME should we update the area here if there are no boxes?
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = targets[i_tgt]["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in targets[0] or "masks" in targets[0]:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        for i_tgt in range(len(targets)):
            if "boxes" in targets[0]:
                cropped_boxes = targets[i_tgt]["boxes"].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = targets[i_tgt]["masks"].flatten(1).any(1)

            for field in fields:
                if field in targets[i_tgt]:
                    targets[i_tgt][field] = targets[i_tgt][field][keep]
    return cropped_clip, targets


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out

    def __call__(self, clip, targets: dict):
        orig_targets = copy.deepcopy(targets)  # used to conserve ALL BOXES ANYWAY
        init_boxes = sum(len(targets[i_tgt]["boxes"]) for i_tgt in range(len(targets)))
        max_patience = 100  # TODO: maybe it is gonna requery lots of time with a clip than an image as it involves more boxes
        for i_patience in range(max_patience):
            if isinstance(clip[0], PIL.Image.Image):
                h = clip[0].height
                w = clip[0].width
            elif isinstance(clip[0], np.ndarray):
                h = clip[0].shape[0]
                w = clip[0].shape[1]
            else:
                raise NotImplementedError
            tw = random.randint(self.min_size, min(w, self.max_size))
            th = random.randint(self.min_size, min(h, self.max_size))

            if h + 1 < th or w + 1 < tw:
                raise ValueError(
                    "Required crop size {} is larger then input image size {}".format(
                        (th, tw), (h, w)
                    )
                )

            if w == tw and h == th:
                region = 0, 0, h, w
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                region = i, j, th, tw

            result_clip, result_targets = crop(clip, targets, region)
            if (not self.respect_boxes) or sum(
                len(result_targets[i_patience]["boxes"])
                for i_patience in range(len(result_targets))
            ) == init_boxes:
                return result_clip, result_targets
            elif self.respect_boxes and i_patience == max_patience - 1:
                # avoid disappearing boxes, targets = result_targets here
                return clip, orig_targets
        return result_clip, result_targets


def make_video_transforms(image_set, cautious, resolution=224):
    """
    :param image_set: train val or test
    :param cautious: whether to preserve bounding box annotations in the spatial random crop
    :param resolution: spatial pixel resolution for the shortest side of each frame
    :return: composition of spatial data transforms to be applied to every frame of a video
    """

    normalizeop = Compose(
        [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    if resolution == 128:
        scales = [96, 128]
        max_size = 213
        resizes = [80, 100, 120]
        crop = 64
        test_size = [128]
    elif resolution == 224:
        scales = [128, 160, 192, 224]
        max_size = 373
        resizes = [100, 150, 200]
        crop = 96
        test_size = [224]
    elif resolution == 256:
        scales = [160, 192, 224, 256]
        max_size = 427
        resizes = [140, 180, 220]
        crop = 128
        test_size = [256]
    elif resolution == 288:
        scales = [160, 192, 224, 256, 288]
        max_size = 480
        resizes = [150, 200, 250]
        crop = 128
        test_size = [288]
    elif resolution == 320:
        scales = [192, 224, 256, 288, 320]
        max_size = 533
        resizes = [200, 240, 280]
        crop = 160
        test_size = [320]
    elif resolution == 352:
        scales = [224, 256, 288, 320, 352]
        max_size = 587
        resizes = [200, 250, 300]
        crop = 192
        test_size = [352]
    elif resolution == 336:
        scales = [224, 256, 288, 320, 352, 336]
        max_size = 640
        resizes = [200, 250, 300]
        crop = 192
        test_size = [384]
    elif resolution == 384:
        scales = [224, 256, 288, 320, 352, 384]
        max_size = 640
        resizes = [200, 250, 300]
        crop = 192
        test_size = [384]
    elif resolution == 416:
        scales = [256, 288, 320, 352, 384, 416]
        max_size = 693
        resizes = [240, 300, 360]
        crop = 224
        test_size = [416]
    elif resolution == 448:
        scales = [256, 288, 320, 352, 384, 416, 448]
        max_size = 746
        resizes = [240, 300, 360]
        crop = 224
        test_size = [448]
    elif resolution == 480:
        scales = [288, 320, 352, 384, 416, 448, 480]
        max_size = 800
        resizes = [240, 300, 360]
        crop = 240
        test_size = [480]
    elif resolution == 800:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        resizes = [400, 500, 600]
        crop = 384
        test_size = [800]
    else:
        raise NotImplementedError

    if image_set == "train":
        horizontal = [] if cautious else [RandomHorizontalFlip()]
        return Compose(
            horizontal
            + [
                RandomSelect(
                    RandomResize(scales, max_size=max_size),
                    Compose(
                        [
                            RandomResize(resizes),
                            RandomSizeCrop(crop, max_size, respect_boxes=cautious),
                            RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalizeop,
            ]
        )

    if image_set in ["val", "test"]:
        return Compose(
            [
                RandomResize(test_size, max_size=max_size),
                normalizeop,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def prepare(w, h, anno):
    """
    :param w: pixel width of the frame
    :param h: pixel height of the frame
    :param anno: dictionary with key bbox
    :return: dictionary with preprocessed keys tensors boxes and orig_size
    """
    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]

    target = {}
    target["boxes"] = boxes
    target["orig_size"] = torch.as_tensor([int(h), int(w)])

    return target