# -*- coding: utf-8 -*-
# @Time    : 2/28/18 8:43 AM

from PIL import Image
import paddle.v2 as paddle
import numpy as np
import os
import sys


class MnistRecognizer():

    def __init__(self):
        # 使用CPU， CPU数量为2
        paddle.init(use_gpu=False, trainer_count=1)

    # 定义卷曲网络模型
    def convolutional_neural_network(self, img):

        # 第一卷积池化层
        conv_pool_1 = paddle.networks.simple_img_conv_pool(input=img,
                                                           filter_size=5,
                                                           num_filters=20,
                                                           num_channel=1,
                                                           pool_size=2,
                                                           pool_stride=2,
                                                           act=paddle.activation.Relu())

        # 第二卷积池化层
        conv_pool_2 = paddle.networks.simple_img_conv_pool(input=conv_pool_1,
                                                           filter_size=5,
                                                           num_filters=50,
                                                           num_channel=20,
                                                           pool_size=2,
                                                           pool_stride=2,
                                                           act=paddle.activation.Relu())

        # 全连接层
        predict = paddle.layer.fc(input=conv_pool_2,
                                  size=10,
                                  act=paddle.activation.Softmax())

        return predict

    # 创建分类器
    def classifier(self):
        images = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(784))

        # 载入之前定义好的网络
        out = self.convolutional_neural_network(images)

        return out

    # 创建训练器
    def trainer(self):

        out = self.classifier()

        # 定义标签
        label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

        # 创建损失函数
        cost = paddle.layer.classification_cost(input=out, label=label)

        # 指定训练相关参数
        parameters = paddle.parameters.create(layers=cost)

        # 定义训练方法
        optimizer = paddle.optimizer.Momentum(learning_rate=0.01/128.0,
                                              momentum=0.9,
                                              regularization=paddle.optimizer.L2Regularization(rate=0.0005*128))

        # 定义训练模型
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)

        return trainer

    # 训练过程
    def training(self):

        # 获取训练器
        trainer = self.trainer()

        lists = []

        # 定义训练事件
        def event_handler(enevt):

            if isinstance(enevt, paddle.event.EndIteration):
                if enevt.batch_id % 100 == 0:
                    print 'Pass %d, Batch %d, Cost %f, %s' % (enevt.pass_id, enevt.batch_id, enevt.cost, enevt.metrics)

            if isinstance(enevt, paddle.event.EndPass):
                with open('%s/models/params_pass_%d.tar' % (os.getcwd(), enevt.pass_id), 'w') as f:
                    trainer.save_parameter_to_tar(f)

                result = trainer.test(reader=paddle.batch(paddle.dataset.mnist.test(), batch_size=128))

                print 'Test with Pass %d, Cost %f, %s\n' % (enevt.pass_id, result.cost, result.metrics)

                lists.append((enevt.pass_id, result.cost, result.metrics['classification_error_evaluator']))

        trainer.train(
            reader=paddle.batch(
                paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=10000),
                batch_size=128),
            event_handler=event_handler,
            num_passes=10)

        # 寻找最小误差的那次训练
        best = sorted(lists, key=lambda x: float(x[1]))[0]
        print 'Best Pass is %s, testing Avgcost is %s' % (best[0], best[1])
        print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)

    # 参数提取器
    def get_parameters(self, n=0):
        with open('%s/models/params_pass_%d.tar' % (os.getcwd(), n), 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters

    # 图像处理器
    def img_handler(self, file=None):

        def load_img(file_path=None):
            img = Image.open(file_path).convert('L')
            img = img.resize((28, 28), Image.ANTIALIAS)
            img = np.array(img).astype(np.float32).flatten()
            img = img / 255.0
            return img

        result = []
        result.append((load_img(file),))
        return result

    # 预测器
    def predicter(self, out_layer, parameter, input_data):

        predict = paddle.infer(output_layer=out_layer,
                               parameters=parameter,
                               input=input_data)

        lab = np.argsort(-predict)
        print 'The Label img is: %d' % lab[0][0]


if __name__ == '__main__':

    MnistReg = MnistRecognizer()
    # # 执行训练
    # MnistReg.training()

    # 预测图片
    img = MnistReg.img_handler('./images/infer_5.png')
    # 选择训练好的模型参数
    parameters = MnistReg.get_parameters(6)

    MnistReg.predicter(MnistReg.classifier(), parameters, img)