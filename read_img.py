import numpy as np
from PIL import Image
from glob import glob

img_dict = {}
for path in glob("data/NVIDIA-10Q-20242905/*"):
    index = path.split('-')[-1][:-4]
    img = Image.open(path)
    img_array = np.asarray(img)
    img_dict[index] = img_array
