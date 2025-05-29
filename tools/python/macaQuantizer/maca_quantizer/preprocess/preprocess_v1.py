import os
import cv2
import torch
import numpy as np
from addict import Addict
from tqdm import trange
from maca_quantizer.utils.utils import get_all_file, maca_warning, read_txt
from maca_quantizer.preprocess import BasePreprocess


class PreprocessV1(BasePreprocess):
   def __init__(self, calib_dir, input_shape, batch_size, calib_num, **kwargs):
      super().__init__(calib_dir, input_shape, batch_size, calib_num)


   def preprocess(self, cfg, device='cuda'):   
      cfg_ = Addict(cfg)
      file_list = []
      if os.path.isfile(self._calib_dir) or self._calib_dir.endswith(".txt"):
         file_list = read_txt(self._calib_dir, cnt=self._calib_num)
      elif os.path.isdir(self._calib_dir):
         file_list = get_all_file(self._calib_dir, cnt=self._calib_num)

      if len(file_list) < self._batch_size*8:
         maca_warning(f'Calibrate dataset number {len(file_list)} too smail, It is better to be more than 8 times batchsize')

      data_tensor = []
      for i in trange(len(file_list), desc="Perprocess "):
         img_path = file_list[i]
         suffix = os.path.splitext(img_path)[1].lstrip(".")
         if suffix.lower() not in {'jpg','bmp','png','jpeg','rgb','tif'}: 
            maca_warning(f"{img_path} is not image")
            continue
         img_tensor = self.process_image(img_path, cfg_.mean, cfg_.std, cfg_.resize, cfg_.isreverse)
         img_tensor = img_tensor.to(device)
         if len(self._input_shape) !=0:
            assert img_tensor.shape == tuple(self._input_shape)
         data_tensor.append(img_tensor)
      return data_tensor

   def process_image(self, fname, img_mean, img_std, resize_setting, isreverse):
      if resize_setting.to[0] == 1:
         image = cv2.imread(fname, flags=0)
      else:
         image = cv2.imread(fname, flags=1)
         if not isreverse and image is not None:
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
   
   @staticmethod
   def letterbox_image(image, new_shape, color=(0, 0, 0)):
      """resize image with unchanged aspect ratio using padding"""
      iw, ih = image.shape[1], image.shape[0]
      w, h = new_shape[1], new_shape[0]

      scale = min(w / iw, h / ih)
      nw = int(iw * scale)
      nh = int(ih * scale)
      new_unpad = (nw, nh)
      img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_CUBIC)

      dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
      dw /= 2  # divide padding into 2 sides
      dh /= 2
      top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
      left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

      new_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
      # cv2.imwrite("tmp.jpg",new_image)
      return new_image