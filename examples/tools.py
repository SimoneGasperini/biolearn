import numpy as np
import pylab as plt


def select_data (dataset, n=None, labels=[]):

  X, y = dataset
  labs = [str(l) for l in labels]
  selected_data = X [ np.isin(y, labs) ]

  return selected_data if n is None else selected_data[:n]


def view_images (images, dims):

  num_images = int(np.sqrt(images.shape[0]))
  selected_images = images[:num_images**2]

  full_image = np.hstack(np.hstack(selected_images.reshape(num_images, num_images, *dims)))

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
  ax.axis('off')
  ax.imshow(full_image, cmap='binary_r')

  plt.show()


def plot_theta(model, outputs=[0]):

  epochs = np.arange(model.num_epochs)
  theta_arr = np.array(model._thetas)

  fig, ax = plt.subplots(figsize=(8,8))

  ax.plot(epochs, theta_arr[:, outputs])

  ax.set_xlabel('epoch', fontsize=18)
  ax.set_ylabel('$\\theta = E[y^2]$', fontsize=18)
  ax.grid()

  plt.show()


def plot_weights_meandiff(model, verbose=True):

  epochs = np.arange(model.num_epochs)
  meandiff = np.array(model._weights_meandiff)

  fig, ax = plt.subplots(figsize=(8,8))

  ax.plot(epochs, meandiff)

  ax.set_xlabel('epoch', fontsize=18)
  ax.set_ylabel('$<W_{diff}>$', fontsize=18)
  ax.grid()

  if verbose:
    ax.set_title(r'$<W_{diff}> = \frac{1}{N} \sum_{ij}\left(\left|W_{t}-W_{t-1}\right|\right)_{ij}$',
             fontsize=18)

  plt.show()
