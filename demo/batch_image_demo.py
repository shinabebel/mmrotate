# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on huge images.

Example:
```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
conda activate openmmlab
python demo/batch_image_demo.py path_to_folder configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth
```
"""  # nowq

from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot

from mmrotate.apis import inference_detector_by_patches

import json
from json import JSONEncoder

import numpy

import glob

class Result:
  def __init__(self, bboxes, labels, classes):
    self.bboxes = bboxes
    self.labels = labels
    self.classes = classes

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_json(ctx, path):
  content = json.dumps(ctx, cls=NumpyArrayEncoder)
  with open(path, "w") as outfile:
    outfile.write(content)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('folder', help='Batch folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    files = glob.glob("{}/*.png".format(args.folder))
    for img in files:
      result = inference_detector_by_patches(model, img, args.patch_sizes, args.patch_steps, args.img_ratios, args.merge_iou_thr)
      bboxes = numpy.vstack(result)
      labels = [ numpy.full(bbox.shape[0], i, dtype=numpy.int32) for i, bbox in enumerate(result) ]
      labels = numpy.concatenate(labels)
      print("img {} get {} results".format(img, bboxes.__len__()))
      name = img.split('.')[0]
      #save_json(bboxes, "{}-bboxes.json".format(name))
      #save_json(labels, "{}-labels.json".format(name))
      #save_json(model.CLASSES, "{}-classes.json".format(name))
      res = Result(bboxes, labels, model.CLASSES)
      content = json.dumps(res.__dict__, cls=NumpyArrayEncoder)
      with open("{}.json".format(name), "w") as outfile:
        outfile.write(content)

      show_result_pyplot(model, img, result, palette=args.palette, score_thr=args.score_thr, out_file="{}-results.jpg".format(name))

if __name__ == '__main__':
    args = parse_args()
    main(args)
