from PIL import Image
from pathlib import Path
import random


data_path = Path('dataset/')
image_path_list = list(data_path.glob("HS-CMU/HS-CMU/image/*.jpg"))
label_path_list = list(data_path.glob("HS-CMU/HS-CMU/label/*.txt"))

def augment_data(image_path_list,label_path_list):
    for i in range(len(image_path_list)):
        name = str(image_path_list[i]).split('\\')[-1].split('.')[0]
        img = Image.open(image_path_list[i])
        with open(label_path_list[i]) as f:
            data = f.read().split()
            label = data[0]
            cx,cy,w,h = data[1:5]

        cx = float(cx)
        cy = float(cy)
        w = float(w)
        h = float(h)
        rand_num = random.randint(1,10)
        image_width, image_height = img.size
        box_width = w * image_width
        box_height = h * image_height
        left = (cx * image_width) - (box_width / 2)
        top = (cy * image_height) - (box_height / 2)
        right = left + box_width
        bottom = top + box_height
        cropped_image = img.crop((left, top, right, bottom))
        if rand_num <=8:
            try:
                cropped_image.save('data/train/{}/{}.jpg'.format(label,name))
            except:
                print('image not saved {}'.format(name))
        elif rand_num >8 and rand_num<=10:
            try:
                cropped_image.save('data/test/{}/{}.jpg'.format(label,name))
            except:
                print('image not saved {}'.format(name))
                
if __name__ == '__main__':
    augment_data(image_path_list=image_path_list,label_path_list=label_path_list)