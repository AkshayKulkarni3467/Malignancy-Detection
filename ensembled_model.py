import torch
import torchvision
import torch.nn as nn

class EnsembledModel:
    def __init__(self):
        self.transforms1,self.model1 = self.make_ResNet_18_finetuned()
        self.transforms2,self.model2 = self.make_Efficient_B0_finetuned()
        self.transforms3,self.model3 = self.make_ViT_B16_finetuned()
    
    @staticmethod  
    def make_ResNet_18_finetuned():
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(in_features=512,out_features=8).to('cpu')
        model.load_state_dict(torch.load(f='models/resnet_18_fine_tuned.pth'))
        return auto_transforms,model

    @staticmethod
    def make_Efficient_B0_finetuned():
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.efficientnet_b0()
        model.classifier = nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,
                  out_features=8)
        ).to('cpu')
        model.load_state_dict(torch.load(f='models/efficientNet_B0_fine_tuned.pth'))
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
        with torch.no_grad():
            output_1 = self.model1(tensor_1)
            output_2 = self.model2(tensor_2)
            output_3 = self.model3(tensor_3)
        # num_1 = torch.argmax(torch.softmax(torch.concat((output_1[0][0:2],torch.tensor([-100]),output_1[0][3:])),0)).cpu().numpy()
        # num_2 = torch.argmax(torch.softmax(torch.concat((output_2[0][0:2],torch.tensor([-100]),output_2[0][3:])),0)).cpu().numpy()
        # num_3 = torch.argmax(torch.softmax(torch.concat((output_3[0][0:2],torch.tensor([-100]),output_3[0][3:])),0)).cpu().numpy()
        num_1 = torch.argmax(torch.softmax(output_1,1)).cpu().numpy()
        num_2 = torch.argmax(torch.softmax(output_2,1)).cpu().numpy()
        num_3 = torch.argmax(torch.softmax(output_3,1)).cpu().numpy()
        output_list = [int(num_1), int(num_2), int(num_3)]  
        # most_common_output = max(set(output_list),key=output_list.count)
        return {'label':int(num_3),'name':category_dict[int(num_3)]}