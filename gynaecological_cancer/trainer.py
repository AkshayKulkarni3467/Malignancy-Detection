from trainers.train_alexnet_model import train_alexnet
from trainers.train_resnet_152_model import train_resnet_152
from trainers.train_convnext_small_model import train_convnext_small
from trainers.train_ViT_B16_model import train_ViT_B16
import torch
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def trainer(epochs,device):
    print('Training ConvNext Small....')
    convnext_small_results = train_convnext_small(epochs,device)
    torch.cuda.empty_cache()
    print('Training AlexNet....')
    alexNet_results = train_alexnet(epochs,device)
    torch.cuda.empty_cache()
    print('Training ResNet....')
    resnet_152_results = train_resnet_152(epochs,device)
    torch.cuda.empty_cache()
    print('Training ViT....')
    ViT_B16_results = train_ViT_B16(epochs,device)
    torch.cuda.empty_cache()
    return alexNet_results,resnet_152_results, convnext_small_results,ViT_B16_results

if __name__ == "__main__":
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Training for {} epochs...'.format(epochs))
    print('Using Device : {}'.format(device))
    output = trainer(epochs,device)
    with open('metrics/ensembled_metrics.pkl','wb') as f:
        pickle.dump(output,f)