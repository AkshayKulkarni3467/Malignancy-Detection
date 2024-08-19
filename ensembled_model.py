import torch
import torchvision
import torch.nn as nn

class EnsembledModel:
    def __init__(self,probs):
        self.w1 = probs[0]
        self.w2 = probs[1]
        self.w3 = probs[2]
        self.w4 = probs[3]
        self.transforms1,self.model1 = self.make_ResNet_152_finetuned()
        self.transforms2,self.model2 = self.make_AlexNet_finetuned()
        self.transforms3,self.model3 = self.make_ViT_B16_finetuned()
        self.transforms4,self.model4 = self.make_ConvNext_finetuned()
        
    
    @staticmethod
    def make_ConvNext_finetuned():
        weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        tranforms = weights.transforms()
        model = torchvision.models.convnext_small()
        model.classifier[2] = nn.Linear(in_features=768,out_features=768,bias=True)
        model.classifier.append(nn.Linear(in_features=768,out_features=8,bias=True))
        model.load_state_dict(torch.load(f='models/ConvNext_small_fine_tuned.pth'))
        return tranforms,model
    
    @staticmethod  
    def make_ResNet_152_finetuned():
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.resnet152()
        model.fc = nn.Linear(in_features=2048,out_features=8,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/resnet_152_fine_tuned.pth'))
        return auto_transforms,model

    @staticmethod
    def make_AlexNet_finetuned():
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        auto_transforms = weights.transforms()
        model = torchvision.models.alexnet()
        model.classifier[4] = nn.Linear(in_features=4096,out_features=4096,bias=True).to('cpu')
        model.classifier[6] = nn.Linear(in_features=4096,out_features=8,bias=True).to('cpu')
        model.load_state_dict(torch.load(f='models/alexnet_fine_tuned.pth'))
        return auto_transforms,model
    
    @staticmethod
    def make_ViT_B16_finetuned():
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        auto_transforms = weights.transforms()
        model = torchvision.models.vit_b_16()
        model.heads = nn.Sequential(
          nn.Linear(in_features=768,out_features=8)
          ).to('cpu')
        model.load_state_dict(torch.load(f='models/ViT_B_16_fine_tuned.pth'))
        return auto_transforms,model
    
    def forward(self,image):
        category_dict = {
        0: "Submucosal Uterine Fibroids",
        1: "Endometrial Cancer",
        2: "Endometrial Polyps",
        3: "Endometrial Polypoid Hyperplasia",
        4: "Endometrial Hyperplasia without Atypia",
        5: "Intrauterine Foreign Body",
        6: "Cervical Polyps",
        7: "Atypical Endometrial Hyperplasia"
    }
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

        return {'label':final_prediction.cpu().item(),'name':category_dict[final_prediction.cpu().item()],'probs' : (p_ensemble/torch.norm(p_ensemble)).cpu().numpy()}