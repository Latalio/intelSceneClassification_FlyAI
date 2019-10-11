# -*- coding: utf-8 -*
import numpy
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
import torch

from path import DATA_PATH

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
device = torch.device('cpu')

class Processor(Base):
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        path = path.replace('\\','/')
        image = Image.open(path)
        image = image.resize((128, 128))
        x_data = numpy.array(image)                     #(width, height, channel)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)    ## scale to [0,1] from [0,255]
        x_data = numpy.transpose(x_data, (2, 0, 1))     ## reshape to (channel, width, height)
        return x_data                                   #(channel, width, height)

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.argmax(data)
