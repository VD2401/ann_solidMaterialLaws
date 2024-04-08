import matplotlib
import matplotlib.pyplot as plt
import torch

OUTPUT_DIR = 'test/test_parameters/output_slice/'
    
# Plot nth slice from 64x64x64 pixel 3D volume

def plot_slice(data, n, axis, save_key, **kwargs):
    data = data.cpu()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    colormap = kwargs.get('cmap', 'jet')
    data_min = kwargs.get('vmin', data.min())
    data_max = kwargs.get('vmax', data.max())
    cmap = plt.get_cmap(colormap, 1024)
    if axis == 0:
        slice = data[n, :, :]
    elif axis == 1:
        slice = data[:, n, :]
    elif axis == 2:
        slice = data[:, :, n]
    mesh = plt.pcolormesh(slice, cmap=cmap, vmin=data_min, vmax=data_max)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    fig.colorbar(mesh)
    plt.savefig(OUTPUT_DIR + 'slice_' + save_key + '.png')
    
def plot_ITOE(input, target, output, n, axis, save_key, **kwargs):
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
    plt.savefig(OUTPUT_DIR + 'input_' + save_key + '.png')
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
    plt.savefig(OUTPUT_DIR + 'target_' + save_key + '.png')
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
    plt.savefig(OUTPUT_DIR + 'output_' + save_key + '.png')
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
    plt.savefig(OUTPUT_DIR + 'error_' + save_key + '.png')
    plt.close()