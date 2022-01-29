
from matplotlib import pyplot


def plot_loss(train_loss_list, valid_loss_list):
    pyplot.figure()
    pyplot.plot(train_loss_list, label="train")
    pyplot.plot(valid_loss_list, label="valid")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.legend()
    pyplot.savefig('graph/_loss.png')
    pyplot.close()
