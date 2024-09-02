from PIL import Image
from pathlib import Path
import random


data_path = Path('dataset/')
image_path_list = list(data_path.glob("*/*.jpeg"))

def augment_data(image_path_list):
    for i in range(len(image_path_list)):
        name,label = str(image_path_list[i]).split('\\')[-1].split('.')[0],str(image_path_list[i]).split('\\')[-2]
        img = Image.open(image_path_list[i])
        rand_num = random.randint(1,10)
        if rand_num <=8:
            try:
                img.save('data/train/{}/{}.jpeg'.format(label,name))
            except:
                print('image not saved {}'.format(name))
        elif rand_num >8 and rand_num<=10:
            try:
                img.save('data/test/{}/{}.jpeg'.format(label,name))
            except:
                print('image not saved {}'.format(name))
                
if __name__ == '__main__':
    augment_data(image_path_list=image_path_list)