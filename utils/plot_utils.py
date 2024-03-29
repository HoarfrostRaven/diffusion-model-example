import torch
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin)/(xmax - xmin)


def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)


def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor([
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [
            0, 0, 0, 0, 1],  [0, 0, 0, 0, 0],      # human, non-human, food, spell, side-facing
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [
            0, 0, 0, 0, 1],  [0, 0, 0, 0, 0],      # human, non-human, food, spell, side-facing
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [
            0, 0, 0, 0, 1],  [0, 0, 0, 0, 0],      # human, non-human, food, spell, side-facing
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [
            0, 0, 0, 0, 1],  [0, 0, 0, 0, 0],      # human, non-human, food, spell, side-facing
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [
            0, 0, 0, 0, 1],  [0, 0, 0, 0, 0],      # human, non-human, food, spell, side-facing
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],  [0, 0, 0, 0, 0]]      # human, non-human, food, spell, side-facing
    )
    return len(vec), vec


def plot_grid(x, n_sample, n_rows, save_dir, w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    # curiously, nrow is number of columns.. or number of items in the row.
    grid = make_grid(norm_torch(x), nrow=ncols)
    save_image(grid, f"{save_dir}run_image_w{w}.png")
    print(f"saved image at {save_dir}run_image_w{w}.png")
    return grid


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    # change to Numpy image format (h,w,channels) vs (channels,h,w)
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)
    # unity norm to put in range [0,1] for np.imshow
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharex=True, sharey=True, figsize=(ncols, nrows))

    def animate_diff(i, store):
        print(f'gif animating frame {i+1} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row*ncols)+col]))
        return plots

    ani = FuncAnimation(fig, animate_diff, fargs=[
                        nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0])
    plt.close()
    if save:
        ani.save(f"{save_dir}{fn}_w{w}.gif",
                 dpi=100, writer=PillowWriter(fps=5))
        print(f"saved gif at {save_dir}{fn}_w{w}.gif")
    return ani


def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4, 2))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()
