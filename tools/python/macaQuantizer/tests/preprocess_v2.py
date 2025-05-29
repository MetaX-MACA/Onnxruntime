
import os
import cv2
import torch
import numpy as np
from addict import Addict
from tqdm import trange
from  maca_quantizer.preprocess import BasePreprocess




class PreprocessV2(BasePreprocess):
    def __init__(self, calib_dir, input_shape, batch_size, calib_num, **kwargs):
        super().__init__(calib_dir, input_shape, batch_size, calib_num)


    def preprocess(self, cfg, devive='cpu'):
        data_tensor = []
        cfg_ = Addict(cfg)
        file_list = os.listdir(self._calib_dir)[:self._calib_num]
        for i in trange(len(file_list), desc="Perprocess "):
            img_path = os.path.join(self._calib_dir, file_list[i])
            img_tensor = self.process_image(img_path, cfg_.mean, cfg_.std, cfg_.resize, cfg_.isreverse)
            if len(self._input_shape) !=0:
                assert img_tensor.shape == tuple(self._input_shape)
            data_tensor.append(img_tensor)
        return data_tensor


    def process_image(self, fname, img_mean, img_std, resize_setting, isreverse):
        if resize_setting.to[0] == 1:
            image = cv2.imread(fname, flags=0)
        else:
            image = cv2.imread(fname, flags=1)
            if not isreverse:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert image is not None, f"Image \'{fname}\' read failed."

        if resize_setting is not None:
            resize_shape = resize_setting.to[1:]
            pad_value = resize_setting.get("pad_value", 0)
            resize_shape.reverse()
            if resize_setting.keep_ratio:
                pad_list = tuple([pad_value] * resize_setting.to[0])
                image = self.letterbox_image(image, tuple(resize_shape), color=pad_list)
            else:
                image = cv2.resize(image, tuple(resize_shape), interpolation=cv2.INTER_CUBIC)
        if 'centercrop' in resize_setting:
            image = self.center_crop(image, resize_setting.centercrop)

        if resize_setting.to[0] == 1:
            image = np.expand_dims(image, axis=-1)
            img_mean = np.mean(img_mean)
            img_std = np.mean(img_std)
        image_norm = np.subtract(image, img_mean)
        image_norm = np.divide(image_norm, img_std, dtype=np.float32)
        image_norm = np.transpose(image_norm, (2, 0, 1))  # hwc-->chw
        img_tensor = torch.from_numpy(image_norm)
        # torch.rand()
        return img_tensor

    @staticmethod
    def center_crop(image, size):
        def crop(image, y, x, h, w):
            return image[y:y + h, x:x + w]

        assert isinstance(image, np.ndarray)
        h, w = image.shape[:2]
        oh, ow = size
        x = int(round((w - ow) / 2.))
        y = int(round((h - oh) / 2.))
        return crop(image, y, x, oh, ow)