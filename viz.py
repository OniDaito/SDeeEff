"""                           
 _____ ___           _____ ___ ___  # noqa
|   __|    \ ___ ___|   __|  _|  _| # noqa
|__   |  |  | -_| -_|   __|  _|  _| # noqa
|_____|____/|___|___|_____|_| |_|   # noqa

Author : Benjamin Blundell - me@benjamin.computer

viz.py - visualise the SDF
"""

# https://zetcode.com/tkinter/drawing/
# https://code.activestate.com/recipes/579048-python-mandelbrot-fractal-with-tkinter/

from tkinter import Tk, Canvas, Frame, PhotoImage, NW, mainloop
import numpy as np
import torch
from loader import TestSDF

RAYSILON = 0.01
MAX_MARCH = 3

class SDFView(Frame):

    def __init__(self, parent, field, rez=128):
        super().__init__(parent)
        self.master.title("SDF")
        self.canvas = Canvas(parent)
        self.rez = rez
        self.img = PhotoImage(width = rez, height = rez)
        self.canvas.create_image((0, 0), image = self.img, state = "normal", anchor = NW)
        self.palette=[ ' #%02x%02x%02x' % (int(255*((i/255)**.25)),0,0) for i in range(256)]
        self.palette.append(' #000000')  #append the color of the centre as index 256
        self.screen_coords = []

        # Coordinates for our rays in world space
        for i in range(self.rez):
            for j in range(self.rez):
                x = float(i) / float(self.rez) * 2.0 - 1.0
                y = float(j) / float(self.rez) * 2.0 - 1.0
                self.screen_coords.append((x, y, i, j))

        # The rastered screen values
        self.raster = []
        for y in range(self.rez):
            self.raster.append([0 for x in range(self.rez)])

        # Raycast
        self.raycast(field)
        self.img.put(self.draw())
        self.canvas.pack()

    def raycast(self, field):
        # assume eye at 0, 0, 0
        
        for coord in self.screen_coords:
            dx = coord[0]
            dy = coord[1]
            sx = coord[2]
            sy = coord[3]
            dz = 0.1 # near plane distance
            ray = np.array([dx, dy, dz])
            norm = np.linalg.norm(ray)
            ray /= norm
            ray = torch.tensor(ray, dtype=torch.float32, device='cpu')
            pos = torch.tensor([dx, dy, dz], dtype=torch.float32, device='cpu')
            d = field.get_distance(pos)

            for i in range(MAX_MARCH):
                pos += ray * d
                d = field.get_distance(pos)
                if d < RAYSILON:
                    # we've hit something
                    self.raster[sy][sx] = 100
                    break
        
    def draw(self):
        return " ".join((("{"+" ".join(self.palette[ self.raster[j][i] ] for i in range(self.rez)))+"}" for j in range(self.rez)))


if __name__ == '__main__':
    rez = 128
    field = TestSDF()
    root = Tk()
    ex = SDFView(root, field, rez=rez)
    root.geometry(str(rez) + "x" + str(rez))
    root.mainloop()