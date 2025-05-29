# coding=utf-8
"""
-------------------------------------------------------------------
   Copyright (c) 2021-2023 Metax Inc. All rights reserved.

   Description :
   create date:   2023/6/13 15:54
-------------------------------------------------------------------
"""


from abc import ABCMeta, abstractmethod



class AbstractPreprocess(metaclass=ABCMeta):
   @abstractmethod
   def preprocess(self, cfg):
      pass


class BasePreprocess(metaclass=ABCMeta):
   def __init__(self, calib_dir, input_shape, batch_size, calib_num):
      self._calib_dir = calib_dir
      self._input_shape = input_shape
      self._batch_size = batch_size
      self._calib_num  = calib_num


   @abstractmethod
   def preprocess(self, cfg):
      raise NotImplementedError('Implement this function first.')
