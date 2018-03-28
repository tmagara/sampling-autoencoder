import argparse

import chainer
from chainer.backends.cuda import get_device_from_id

import visualize
from model import SamplingVAE


def main():
    parser = argparse.ArgumentParser(description='Resolution independent image representation on neural network.')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = SamplingVAE()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(True, 3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, False, False)
    dump_iter = chainer.iterators.SerialIterator(test, 16)

    stop_trigger = (args.epoch, 'epoch')

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, stop_trigger, out=args.out)
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(chainer.training.extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'
    ), trigger=(10, 'epoch'))

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch',
         'main/x_loss', 'validation/main/x_loss',
         'main/z_loss',
         'elapsed_time']))

    trainer.extend(visualize.Visualize(dump_iter, model, device=args.gpu))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
