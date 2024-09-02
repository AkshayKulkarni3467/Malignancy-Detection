from PIL import Image
from ensembled_model import EnsembledModel
import matplotlib.pyplot as plt
from pathlib import Path
import time
import random

def counter_to_check(data_path,probs):
    model = EnsembledModel(probs)
    data_path = Path(data_path)
    image_list = list(data_path.glob('*/*.jpeg'))
    random.shuffle(image_list)
    for i in image_list:
        real_label = str(i).split('\\')[-2]
        img = Image.open(i)
        label_dict = model.forward(img)
        print('--------------------------------')
        print(f'''{{Original label : {real_label}}}''')
        print(label_dict)
        time.sleep(0.5)
        
def predict(image_path,probs):
    model = EnsembledModel(probs)
    img = Image.open(image_path)
    label_dict = model.forward(img)
    print('--------------------------------')
    print(label_dict)
    return label_dict

if __name__ == "__main__":
    counter_to_check('data/train',[0.25,0.25,0.25,0.25])