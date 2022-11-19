import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.__mult_coef = int(self.resolution / self.tile_size / 2)
        self.output = None

    def draw(self):
        if self.resolution % (2*self.tile_size):
            return None
        black_tile = np.tile(np.array([0]), (self.tile_size, self.tile_size))
        white_tile = np.tile(np.array([1]), (self.tile_size, self.tile_size))
        basic_tile = np.vstack((np.hstack((black_tile, white_tile)), np.hstack((white_tile, black_tile))))
        self.output = np.tile(basic_tile, (self.__mult_coef, self.__mult_coef))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        h, k = self.position
        length = np.arange(self.resolution)
        width = np.arange(self.resolution)
        x, y = np.meshgrid(length, width)
        board = (x - h) ** 2 + (y - k) ** 2
        self.output = (board <= self.radius ** 2).astype(int)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros([self.resolution, self.resolution, 3])
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()
