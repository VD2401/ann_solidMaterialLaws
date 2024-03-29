import matplotlib
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

def learning(training_indicator, testing_indicator, training_time, **kwargs):
    
    data_min = kwargs.get('vmin', 0)
    data_max = kwargs.get('vmax', max([max(training_indicator), max(testing_indicator)]))

    plt.figure(figsize=(10, 6), dpi=200)

    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.plot(range(1, len(testing_indicator) + 1), training_indicator, label='Training error', linewidth=1, markersize=7, marker='+')
    plt.plot(range(1, len(testing_indicator) + 1), testing_indicator, label='Testing error', linewidth=1, markersize=7, marker='+')

    plt.xlabel('Epochs')
    plt.xlim([0, len(testing_indicator) + 1])

    plt.ylabel('Mean Absolute Error')
    plt.ylim([data_min, data_max])

    plt.grid()
    plt.legend()
    plt.annotate('Epoch duration = ' + str(round(training_time, 2)) + ' s',
                 (0.05 * (len(testing_indicator) + 1), 0.95 * max([max(training_indicator), max(testing_indicator)])))

    # plt.yscale('log')

    plt.show()
    clear_output(wait=True)
    

def surface(data, **kwargs):

    data = data.cpu()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    
    colormap = kwargs.get('cmap', 'jet')
    data_min = kwargs.get('vmin', data.min())
    data_max = kwargs.get('vmax', data.max())

    cmap = plt.get_cmap(colormap, 1024)

    mesh = plt.pcolormesh(torch.arange(data.shape[0]), torch.arange(data.shape[1]), data.transpose(0, 1), cmap=cmap, vmin=data_min, vmax=data_max)
    
    # ax.set(xlabel='X', ylabel='Y')

    ax.set_aspect('equal')
    fig.colorbar(mesh)

    # fig.savefig(kwargs.get('output', 'output') + '.png')
    plt.show()


def volume(x, y, z, data, **kwargs):

    x = x.cpu()
    y = y.cpu()
    z = z.cpu()
    data = data.cpu()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    x = x - dx / 2
    x = x.add(x[-1] + dx)
    y = y - dy / 2
    y = y.add(y[-1] + dy)
    z = z - dz / 2
    z = z.add(z[-1] + dz)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    colormap = kwargs.get('cmap', 'jet')
    data_min = kwargs.get('vmin', data.min())
    data_max = kwargs.get('vmax', data.max())

    norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)
    data = norm(data)

    args = {'linewidth': 0, 'antialiased': False, 'rstride': 1, 'cstride': 1, 'shade': False}
    cmap = plt.get_cmap(colormap, 1024)

    my_col = cmap.__call__(data[:, :, -1])
    ax.plot_surface(X[:, :, -1], Y[:, :, -1], Z[:, :, -1], facecolors=my_col, **args)

    my_col = cmap.__call__(data[:, 0, :])
    ax.plot_surface(X[:, 0, :], Y[:, 0, :], Z[:, 0, :], facecolors=my_col, **args)

    my_col = cmap.__call__(data[0, :, :])
    ax.plot_surface(X[0, :, :], Y[0, :, :], Z[0, :, :], facecolors=my_col, **args)

    # my_col = cmap.__call__(data[:, :, 0])
    # ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0], facecolors=my_col, **args)

    # my_col = cmap.__call__(data[:, -1, :])
    # ax.plot_surface(X[:, -1, :], Y[:, -1, :], Z[:, -1, :], facecolors=my_col, **args)

    # my_col = cmap.__call__(data[-1, :, :])
    # test = ax.plot_surface(X[-1, :, :], Y[-1, :, :], Z[-1, :, :], facecolors=my_col, **args)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()

    # edges_kw = dict(color='k', linewidth=1, zorder=1e3)
    # ax.plot([xmin, xmax], [ymin, ymin], zmin, **edges_kw)
    # # ax.plot([xmin, xmax], [ymax, ymax], zmin, **edges_kw)
    # ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
    # ax.plot([xmin, xmax], [ymax, ymax], zmax, **edges_kw)

    # ax.plot([xmin, xmin], [ymin, ymax], zmin, **edges_kw)
    # # ax.plot([xmax, xmax], [ymin, ymax], zmin, **edges_kw)
    # ax.plot([xmin, xmin], [ymin, ymax], zmax, **edges_kw)
    # ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)

    # ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    # ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
    # ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)
    # # ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)

    ax.set(xlabel='X', ylabel='Y', zlabel='Z')

    ax.view_init(15, -105, 0)
    
    ax.set(xlim=[x.min(), x.max()], ylim=[y.min(), y.max()], zlim=[z.min(), z.max()])
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax)
    
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')

    plt.show()