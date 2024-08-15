from PIL import Image
from ensembled_model import EnsembledModel
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random

def counter_to_check(data_path):
    model = EnsembledModel()
    data_path = Path(data_path)
    image_list = list(data_path.glob('*/*.jpg'))
    random.shuffle(image_list)
    for i in image_list:
        real_label = str(i).split('\\')[-2]
        img = Image.open(i)
        label_dict = model.forward(img)
        print('--------------------------------')
        print(f'''{{Original label : {real_label}}}''')
        print(label_dict)
        time.sleep(0.5)
        
def predict(image_path):
    model = EnsembledModel()
    img = Image.open(image_path)
    label_dict = model.forward(img)
    print('--------------------------------')
    print(label_dict)
    return label_dict

if __name__ == "__main__":
    predict('data/train/4/EH001_806.jpg')