import mlflow

import torch
import matplotlib
import matplotlib.pyplot as plt

OUTPUT_IMAGES_PATH = "output_images/"
def learning(training_indicator, testing_indicator, training_time, **kwargs):
    
    data_min = kwargs.get('vmin', 0)
    data_max = kwargs.get('vmax', max([max(training_indicator), max(testing_indicator)]))

    fig = plt.figure(figsize=(10, 6), dpi=200)

    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.plot(range(1, len(testing_indicator) + 1), training_indicator, label='Training error', linewidth=1, markersize=7, marker='+')
    plt.plot(range(1, len(testing_indicator) + 1), testing_indicator, label='Testing error', linewidth=1, markersize=7, marker='+')

    plt.xlabel('Epochs')
    plt.xlim([0, len(testing_indicator) + 1])

    plt.yscale('log')
    plt.ylabel('Mean Absolute Error')
    plt.ylim([data_min, data_max])

    plt.grid()
    plt.legend()
    plt.annotate('Epoch duration = ' + str(round(training_time, 2)) + ' s',
                 (0.05 * (len(testing_indicator) + 1), 0.95 * max([max(training_indicator), max(testing_indicator)])))
    # save figure
    print(OUTPUT_IMAGES_PATH + 'training.png')
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + 'training.png')
    
    plt.close()
# %%

def plot_ITOE(input, target, output, n, axis, epoch, **kwargs):
    # Plot nth slice from 64x64x64 pixel 3D volume for input, target and output with same color scale for output and target
    input = input.cpu()
    target = target.cpu()
    output = output.cpu()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    colormap = kwargs.get('cmap', 'jet')
    input_min = kwargs.get('vmin', input.min())
    input_max = kwargs.get('vmax', input.max())
    cmap = plt.get_cmap(colormap, 1024)
    if axis == 0:
        slice = input[n, :, :]
    elif axis == 1:
        slice = input[:, n, :]
    elif axis == 2:
        slice = input[:, :, n]
    mesh = plt.pcolormesh(slice, cmap=cmap, vmin=input_min, vmax=input_max)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + 'input@ep' + str(epoch) + '.png')
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    colormap = kwargs.get('cmap', 'jet')
    cross_min = kwargs.get('vmin', min(target.min(), output.min()))
    cross_max = kwargs.get('vmax', max(target.max(), output.max()))
    cmap = plt.get_cmap(colormap, 1024)
    if axis == 0:
        slice = target[n, :, :]
    elif axis == 1:
        slice = target[:, n, :]
    elif axis == 2:
        slice = target[:, :, n]
    mesh = plt.pcolormesh(slice, cmap=cmap, vmin=cross_min, vmax=cross_max)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + 'target@ep' + str(epoch) + '.png')
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    colormap = kwargs.get('cmap', 'jet')
    if axis == 0:
        slice = output[n, :, :]
    elif axis == 1:
        slice = output[:, n, :]
    elif axis == 2:
        slice = output[:, :, n]
    mesh = plt.pcolormesh(slice, cmap=cmap, vmin=cross_min, vmax=cross_max)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + 'output@ep' + str(epoch) + '.png')
    
    # plot error between target and output
    error = (target - output).abs().cpu()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    colormap = kwargs.get('cmap', 'jet')
    error_min = kwargs.get('vmin', error.min())
    error_max = kwargs.get('vmax', error.max())
    cmap = plt.get_cmap(colormap, 1024)
    if axis == 0:
        slice = error[n, :, :]
    elif axis == 1:
        slice = error[:, n, :]
    elif axis == 2:
        slice = error[:, :, n]
    mesh = plt.pcolormesh(slice, cmap=cmap, vmin=error_min, vmax=error_max)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + 'MAE@ep' + str(epoch) + '.png')
    plt.close()

def volume(x, y, z, data, data_type: str, epoch, **kwargs):
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
    
    fig.tight_layout()
    
    print(OUTPUT_IMAGES_PATH + data_type +'@ep' + str(epoch) + '.png')
    mlflow.log_figure(fig, OUTPUT_IMAGES_PATH + data_type +'@ep' + str(epoch) + '.png')
    
    plt.close()