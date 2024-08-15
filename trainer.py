from trainers.train_efficient_B0_model import train_efficientNet_B0
from trainers.train_resnet_18_model import train_resnet_18
from trainers.train_TinyVGG_base_model import train_TinyVGG
from trainers.train_ViT_B16_model import train_ViT_B16
import torch
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def trainer(epochs,device):
    print('Training TinyVGG....')
    tinyVGG_results = train_TinyVGG(epochs,device)
    torch.cuda.empty_cache()
    print('Training EfficientNet....')
    efficientNet_B0_results = train_efficientNet_B0(epochs,device)
    torch.cuda.empty_cache()
    print('Training ResNet....')
    resnet_18_results = train_resnet_18(epochs,device)
    torch.cuda.empty_cache()
    print('Training ViT....')
    ViT_B16_results = train_ViT_B16(epochs,device)
    torch.cuda.empty_cache()
    return efficientNet_B0_results, resnet_18_results, tinyVGG_results,ViT_B16_results

if __name__ == "__main__":
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Training for {} epochs...'.format(epochs))
    print('Using Device : {}'.format(device))
    output = trainer(epochs,device)
    with open('metrics/ensembled_metrics.pkl','wb') as f:
        pickle.dump(output,f)