import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


"""
Basically copy-pasted from this example:
https://matplotlib.org/stable/gallery/animation/dynamic_image.html#sphx-glr-gallery-animation-dynamic-image-py
"""

fig, ax = plt.subplots(figsize=(10,10), dpi=300)
ims = []

# go-go-gadget copycat:
# https://stackoverflow.com/questions/50629968/how-to-sort-files-by-number-in-filename-of-a-directory
files = os.listdir('animation')
files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

for i, file in enumerate(files):
    arr = np.loadtxt('animation/' + file, delimiter=',')
    im = ax.imshow(arr, vmin=0, vmax=1, interpolation='none', animated=True)
    if i == 0:
        ax.imshow(arr, vmin=0, vmax=1, interpolation='none')  # show an initial one first
    ims.append([im])

anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
anim.save("animation.mp4")

#plt.show()
