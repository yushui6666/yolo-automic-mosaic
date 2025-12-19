import colorsys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


def mosaic_area(img_bgr, left, top, right, bottom, block=14):
    """
    对 OpenCV BGR 图像的指定区域打马赛克（像素化）
    block: 马赛克块大小，越大越糊（脱敏更强）
    """
    h, w = img_bgr.shape[:2]
    left = max(0, min(int(left), w - 1))
    right = max(0, min(int(right), w))
    top = max(0, min(int(top), h - 1))
    bottom = max(0, min(int(bottom), h))

    if right <= left or bottom <= top:
        return img_bgr

    roi = img_bgr[top:bottom, left:right]
    if roi.size == 0:
        return img_bgr

    rh, rw = roi.shape[:2]
    sw = max(1, rw // block)
    sh = max(1, rh // block)

    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img_bgr[top:bottom, left:right] = mosaic
    return img_bgr


class YOLO(object):
    _defaults = {
        "model_path": 'model/best_epoch_weights.pth',
        "classes_path": 'model/voc_classes.txt',
        "input_shape": [640, 640],
        "phi": 's',
        "confidence": 0.3,
        "nms_iou": 0.3,
        "letterbox_image": True,
        "cuda": True,

        # 脱敏参数（可按需调整）
        "mosaic_block": 14,      # 马赛克强度：越大越糊
        "expand_ratio": 0.2,    # 扩框比例：防漏遮
        "min_face_size": 10,     # 太小的框不处理（像素）
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = YoloBody(self.input_shape, self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def _infer(self, image):
        """
        内部推理：返回 results[0] 或 None
        results[0] 每行: [top, left, bottom, right, conf, cls]
        """
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        if results[0] is None:
            return None
        return results[0]

    def detect_boxes(self, image):
        """
        给跟踪用：只输出检测框（xyxy）、score、label
        返回：
          det_xyxy: (N,4) float32  [x1,y1,x2,y2]
          det_scores: (N,) float32
          det_labels: (N,) int32
        """
        r = self._infer(image)
        if r is None:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))

        top_label = np.array(r[:, 5], dtype='int32')
        top_conf = r[:, 4].astype(np.float32)
        top_boxes = r[:, :4].astype(np.float32)  # [top,left,bottom,right]

        # 转为 xyxy: [x1,y1,x2,y2]
        det_xyxy = np.stack([top_boxes[:, 1], top_boxes[:, 0], top_boxes[:, 3], top_boxes[:, 2]], axis=1).astype(np.float32)
        return det_xyxy, top_conf, top_label

    def detect_image(self, image, crop=False, count=False):
        """
        单张图片：直接输出“马赛克脱敏”后的 PIL.Image（RGB）
        """
        r = self._infer(image)
        image = cvtColor(image)

        if r is None:
            return image

        top_label = np.array(r[:, 5], dtype='int32')
        top_conf = r[:, 4].astype(np.float32)
        top_boxes = r[:, :4].astype(np.float32)  # [top,left,bottom,right]

        if count:
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, _ in enumerate(top_boxes):
                top, left, bottom, right = top_boxes[i]
                top = max(0, int(top))
                left = max(0, int(left))
                bottom = min(image.size[1], int(bottom))
                right = min(image.size[0], int(right))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)

        # PIL -> OpenCV(BGR) 一次
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        w_img, h_img = image.size

        for i, c in list(enumerate(top_label)):
            score = float(top_conf[i])
            box = top_boxes[i]
            top, left, bottom, right = box
            top = max(0, int(top))
            left = max(0, int(left))
            bottom = min(h_img, int(bottom))
            right = min(w_img, int(right))

            bw = right - left
            bh = bottom - top
            if bw < self.min_face_size or bh < self.min_face_size:
                continue

            # 扩框
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            nw = int(bw * (1 + self.expand_ratio))
            nh = int(bh * (1 + self.expand_ratio))
            left_e = max(0, cx - nw // 2)
            right_e = min(w_img, cx + nw // 2)
            top_e = max(0, cy - nh // 2)
            bottom_e = min(h_img, cy + nh // 2)

            image_cv = mosaic_area(image_cv, left_e, top_e, right_e, bottom_e, block=self.mosaic_block)

        # OpenCV -> PIL(RGB)
        image_out = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        return image_out

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            _ = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape,
                image_shape, self.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                _ = self.bbox_util.non_max_suppression(
                    outputs, self.num_classes, self.input_shape,
                    image_shape, self.letterbox_image,
                    conf_thres=self.confidence, nms_thres=self.nms_iou
                )
        t2 = time.time()
        return (t2 - t1) / test_interval
