import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers
import numpy as np
import _pickle as cpickle
import os
import random

professors = {
	'ito': 0,
	'inamura': 1,
	'ushiama': 2,
	'kimu': 3,
	'takagi': 4,
	'ueoka': 5,
	'oshima': 6,
	'tomotani': 7,
	'tsuruno': 8,
	'fuyuno': 9,
	'fujihara': 10,
	'tomimatsu': 11
}


def load_pickle():
    face_path = 'af_face_pickle'
    imageData = []
    labelData = []
    test_tuple = []
    af_imageData = []
    af_labelData = []
    for directory in os.listdir(face_path):
        if '.DS_Store' in directory:
            continue
        file_path = face_path + '/' + directory
        for img in os.listdir(file_path):
            if '.DS_Store' in img or '_DS_Store' in img:
                continue
            pickle_dt = np.load(file_path + '/' + img).reshape(3, 5625)
            r, g, b = pickle_dt[0], pickle_dt[1], pickle_dt[2]
            rImg = np.asarray(np.float32(r) / 255.0).reshape(75, 75)
            gImg = np.asarray(np.float32(g) / 255.0).reshape(75, 75)
            bImg = np.asarray(np.float32(b) / 255.0).reshape(75, 75)
            all_ary = np.asarray([rImg, gImg, bImg])
            test_tuple.append((all_ary, np.int32(int(directory))))
            imageData.append(all_ary)
            labelData.append(np.int32(int(directory)))

    random.shuffle(test_tuple)

    for tuple in test_tuple:
        af_imageData.append(tuple[0])
        af_labelData.append(tuple[1])

    return af_imageData, af_labelData


class SimpleAlex(chainer.Chain):
	def __init__(self):
		super(SimpleAlex, self).__init__(
			conv1 = L.Convolution2D(None, 50, 6, stride=3),
			conv2 = L.Convolution2D(None, 75, 3, pad=1),
			fc1 = L.Linear(None, 200),
			fc2 = L.Linear(None, 11)
			)

	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=1)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=1)
		h = F.dropout(F.relu(self.fc1(h)))
		h = self.fc2(h)

		return h


# モデルのインスタンス化
model = SimpleAlex()
classify_model = L.Classifier(model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(classify_model)


imageData, labelData = load_pickle()
threshold = np.int32(len(imageData)/5*4)
train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])

train_iter = chainer.iterators.SerialIterator(train, 100, shuffle=True)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (40, 'epoch'), out='Geijo_result')
trainer.extend(extensions.Evaluator(test_iter, classify_model, device=0))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(y_keys='main/loss', file_name='loss.png'))
trainer.extend(extensions.PlotReport(y_keys='main/accuracy', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/loss', file_name='overfitting.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/accuracy', file_name='validation_acc.png'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()

serializers.save_npz('geijo_result.npz', model)
