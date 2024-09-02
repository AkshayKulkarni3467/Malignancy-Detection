import torch
import torchvision
import torch.nn as nn
from data_labels import data_labels

class EnsembledModel:
    def __init__(self,probs):
        self.w1 = probs[0]
        self.w2 = probs[1]
        self.w3 = probs[2]
        self.w4 = probs[3]
        self.transforms1,self.model1 = self.make_RegNet_X_finetuned()
        self.transforms2,self.model2 = self.make_RegNet_Y_finetuned()
        self.transforms3,self.model3 = self.make_Swin_B_finetuned()
        self.transforms4,self.model4 = self.make_VGG_11_finetuned()
        
    
    @staticmethod
    def make_RegNet_X_finetuned():
        weights = torchvision.models.RegNet_X_16GF_Weights.IMAGENET1K_V1
        tranforms = weights.transforms()
        model = torchvision.models.regnet_x_16gf()
        model.fc = nn.Linear(in_features=2048,out_features=5,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/regnet_x_fine_tuned.pth'))
        return tranforms,model
    
    @staticmethod  
    def make_RegNet_Y_finetuned():
        weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_V1
        auto_transforms = weights.transforms()
        model = torchvision.models.regnet_y_16gf()
        model.fc = nn.Linear(in_features=3024,out_features=5,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/regnet_y_fine_tuned.pth'))
        return auto_transforms,model

    @staticmethod
    def make_Swin_B_finetuned():
        weights = torchvision.models.Swin_B_Weights.IMAGENET1K_V1
        auto_transforms = weights.transforms()
        model = torchvision.models.swin_b()
        model.head = nn.Linear(in_features=1024,out_features=5,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/swin_b_fine_tuned.pth'))
        return auto_transforms,model
    
    @staticmethod
    def make_VGG_11_finetuned():
        weights = torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1
        auto_transforms = weights.transforms()
        model = torchvision.models.vgg11_bn()
        model.classifier[3] = nn.Linear(in_features=4096,out_features=4096,bias=True).to('cpu')
        model.classifier[6] = nn.Linear(in_features=4096,out_features=5,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/vgg11_fine_tuned.pth'))
        return auto_transforms,model
    
    def forward(self,image):
        tensor_1 = self.transforms1(image)
        tensor_1 = tensor_1.view(1,*(tensor_1.shape))
        tensor_2 = self.transforms2(image)
        tensor_2 = tensor_2.view(1,*(tensor_2.shape))
        tensor_3 = self.transforms3(image)
        tensor_3 = tensor_3.view(1,*(tensor_3.shape))
        tensor_4 = self.transforms4(image)
        tensor_4 = tensor_4.view(1,*(tensor_4.shape))
        with torch.no_grad():
            output_1 = self.model1(tensor_1)
            output_2 = self.model2(tensor_2)
            output_3 = self.model3(tensor_3)
            output_4 = self.model4(tensor_4)
        
        num_1_array = torch.softmax(output_1,1)
        num_2_array = torch.softmax(output_2,1)
        num_3_array = torch.softmax(output_3,1)
        num_4_array = torch.softmax(output_4,1)
        
        p_ensemble = self.w1 * num_1_array + self.w2 * num_2_array + self.w3 * num_3_array + self.w4 * num_4_array
        
        final_prediction = torch.argmax(p_ensemble)

        return {'label':final_prediction.cpu().item(),'name':data_labels[final_prediction.cpu().item()],'probs' : (p_ensemble/torch.norm(p_ensemble)).cpu().numpy()}