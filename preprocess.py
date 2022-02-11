from os import chdir, listdir, mkdir
import torchvision.transforms.functional as F 
from PIL import Image 
from collections import defaultdict

chdir('img')
counter = 0

for d in listdir():
	chdir(d)
	category = d.split('_')[-1]
	print(category)
	for file in listdir():
		with Image.open(file) as im:
			t = F.center_crop(im, min(im.size))
			t = F.resize(t, 128)
			t.save(f'../../small/{category}/{counter}.jpg')
			counter += 1
	chdir('..')