from trainers.train_regnet_x_model import train_regnet_x
from trainers.train_regnet_y_model import train_regnet_y
from trainers.train_swin_b_model import train_swin_b
from trainers.train_vgg11_model import train_vgg11
import torch
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def trainer(epochs,device):
    print('Training RegNet X....')
    regnet_x_results = train_regnet_x(epochs,device)
    torch.cuda.empty_cache()
    print('Training RegNet Y....')
    regnet_y_results = train_regnet_y(epochs,device)
    torch.cuda.empty_cache()
    print('Training Swin B....')
    swin_b_results = train_swin_b(epochs,device)
    torch.cuda.empty_cache()
    print('Training VGG 11....')
    vgg11_results = train_vgg11(epochs,device)
    torch.cuda.empty_cache()
    return regnet_x_results,regnet_y_results,swin_b_results,vgg11_results

if __name__ == "__main__":
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Training for {} epochs...'.format(epochs))
    print('Using Device : {}'.format(device))
    output = trainer(epochs,device)
    with open('metrics/ensembled_metrics.pkl','wb') as f:
        pickle.dump(output,f)