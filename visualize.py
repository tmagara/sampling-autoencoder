import pathlib

import chainer
from chainer.backends.cuda import to_cpu
from matplotlib import pyplot


class Visualize(chainer.training.Extension):
    trigger = (1, 'epoch')

    def __init__(self, iterator, target, converter=chainer.dataset.convert.concat_examples, device=None):
        self.iterator = iterator
        self.target = target
        self.converter = converter
        self.device = device

    def __call__(self, trainer):
        x, labels = self.converter(self.iterator.next(), self.device)
        images_map = self.target.show(x)
        for key, images in images_map.items():
            images = numpy2pyplot(images)
            path = pathlib.Path(trainer.out) / pathlib.Path("epoch_{}_{}.png".format(trainer.updater.epoch, key))
            save_cifar_vh(images, str(path), 1)


def numpy2pyplot(images):
    images = chainer.functions.broadcast_to(images, (images.shape[0], 3) + images.shape[2:])
    images = chainer.functions.transpose(images, (0, 2, 3, 1))
    images = chainer.functions.reshape(images, (4, -1) + images.shape[1:])
    images = chainer.backends.cuda.to_cpu(images.data)
    return images


def save_cifar_vh(images, filepath, dpi=100):
    rows, columns, h, w, _ = images.shape
    pyplot.figure(figsize=(w * columns / dpi, h * rows / dpi), dpi=dpi)
    for r in range(0, rows):
        for c in range(0, columns):
            pyplot.subplot(rows, columns, r * columns + c + 1)
            pyplot.imshow(images[r, c], interpolation="nearest")
            pyplot.axis('off')
            pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    pyplot.savefig(filepath, dpi=dpi, facecolor='black')
    pyplot.close()
