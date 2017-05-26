#!/usr/bin/python3
"""
Author: Jiaqing Lin
E-mail: Jiaqing930@gmail.com
This program is used to train temporal stream model.
"""

import read_data
from chainer import training, iterators, optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class CNN_Temporal(Chain):
    """
    Input frame size is (20 x 224 x224).
    4 class labels in this dataset.
    """
    def __init__(self, n_label=4):
        super(CNN_Temporal, self).__init__(
            conv1=L.Convolution2D(in_channels=20, out_channels=96, ksize=7, stride=2),
            conv2=L.Convolution2D(in_channels=96, out_channels=256, ksize=5, stride=2),
            conv3=L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1),
            conv4=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            conv5=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1),
            fc6=L.Linear(in_size=None, out_size=4096),
            fc7=L.Linear(in_size=4096, out_size=2048),
            fc8=L.Linear(in_size=2048, out_size=n_label)
        )


    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = F.relu(self.conv5(h))
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.9)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.8)
        h = self.fc8(h)

        return h


if __name__ == '__main__':

    print('===== << Start training temporal_model... >> =====')
    print('Minibatch-size: {0}, epch: {1}'.format(100, 20))

    # 1. Setup model.
    model = CNN_Temporal()
    classifier_model = L.Classifier(model)

    # 2. Setup optimizer.
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.use_cleargrads()
    optimizer.setup(classifier_model)

    # 3. Load dataset.
    train, test = read_data.temporal_dataset()

    # 4. Setup iterator.
    train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    # 5. Setup updater.
    updater = training.StandardUpdater(train_iter, optimizer)

    # 6. Setup trainer
    trainer = training.Trainer(updater, (20, 'epoch'), out='temporal_result')

    # Evaluate the model with test dataset for each epoch.
    trainer.extend(extensions.Evaluator(test_iter, classifier_model))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
