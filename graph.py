
from matplotlib import pyplot


def plot(source, output, target, epoch):
    pyplot.plot(source, color="green", label="source", linestyle="dashed", marker='o')
    pyplot.plot(target, color="blue", label="target")
    pyplot.plot(output, color="red", label="output")
    pyplot.grid(True, which='both')
    pyplot.legend()
    pyplot.savefig(f'graph/train-epoch{epoch}.png')
    pyplot.close()


def plot_name(source, output, target, name):
    pyplot.plot(source, color="green", label="source", linestyle="dashed", marker='o')
    pyplot.plot(target, color="blue", label="target")
    pyplot.plot(output, color="red", label="output")
    pyplot.grid(True, which='both')
    pyplot.legend()
    pyplot.savefig(f'graph/{name}.png')
    pyplot.close()


def plot_loss(train_loss_list, valid_loss_list):
    pyplot.figure()
    pyplot.plot(train_loss_list, label="train")
    pyplot.plot(valid_loss_list, label="valid")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.ylim(0.0, 0.5)
    pyplot.legend()
    pyplot.savefig('graph/_transformer-loss.png')
    pyplot.close()
