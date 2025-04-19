import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
import random
import matplotlib
matplotlib.use('TkAgg')

class animator():
    
    def use_svg_display(self):
        backend_inline.set_matplotlib_formats('svg')

    def set_figsize(self, figsize = (3.5, 2.5)):
        self.use_svg_display()
        plt.rcParams['figure.figsize'] = figsize

    def set_axes(self):
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        if self.legend:
            self.axes.legend(self.legend)
        self.axes.grid()

    def __init__(self, 
             xlabel = None, xlim = None, xscale = 'linear',
             ylabel = None, ylim = None, yscale = 'linear',
             legend = None,
             fmts = ('-', 'm--', 'g-.', 'r:'),
             nrows = 1, ncols = 1, figsize = (3.5, 2.5) ):
        if legend is None:
            legend = []
        self.xlabel, self.xlim, self.xscale = xlabel, xlim, xscale
        self.ylabel, self.ylim, self.yscale = ylabel, ylim, yscale
        self.legend = legend

        self.use_svg_display()
        self.fig, self.axes = plt.subplots()

        self.set_axes()
         
        self.X, self.Y, self.fmts = None, None, fmts
    
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        self.axes.cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        self.set_axes()
        display.display(self.fig)

        plt.draw()
        plt.pause(0.01)

        display.clear_output(wait=True)
    
    def show(self):
        plt.show()


def main():
    Animator = animator(
        xlabel = 'epoch', xlim = [1, 20], legend = ['train loss', 'train acc', 'test acc']
    )
    
    for i in range(20):
        test_acc = random.random()
        train_acc = random.random()
        train_l = random.random()

        Animator.add(i + (i + 1) / 20, (train_l, train_acc, None))
        Animator.add(i + 1, (None, None, test_acc))


if __name__ == '__main__':
    main()
