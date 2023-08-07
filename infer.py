import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Old weights with accuracy 76.130%
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load('./checkpoints/2.pth'))
model.to('cuda')
import os
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

import torchvision.transforms as transforms
transform  = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),])

class_dict = {0:'axial', 1:'coronal'}


softmax = nn.Softmax(dim=-1)
users = [user for user in os.listdir('./dataset') if os.path.isdir(f'./dataset/{user}')]
with torch.no_grad():
    for subjectID in tqdm(users[394:395]):
        #subjectID = f'AJA{i:04}'
        path      = f'./dataset/{subjectID}/image/{subjectID}'
        dest      = {0:f'./dataset/{subjectID}/image/axial',
                     1:f'./dataset/{subjectID}/image/coronal'}
        for i in range(2):
            os.makedirs(dest[i], exist_ok=True)
        files     = [file for file in os.listdir(path) if file.endswith('.tif')]
        
        files.sort()

        files     = files[1:]   #remove sagittal image this may remove non-sagittal frame but let's omit this
        # print(f'Subject ID: {subjectID}')
        # print(f'Number of files: {len(files)}')
        imgs = torch.zeros((len(files),3,224,224))
        for idx, file in enumerate(files):
            imgs[idx] = transform(Image.open(f'{path}/{file}'))
        imgs  = imgs.to('cuda')
        logit = model(imgs)
        pred  = torch.argmax(logit, dim=-1)
        prob  = torch.max(softmax(logit), dim=-1)
        prob  = prob[0].cpu().numpy()
        pred  = pred.detach().cpu().numpy()
        for idx, p in enumerate(pred):
            # print(f'{files[idx]}: {class_dict[p]:>10} {prob[idx]*100:.2f}%')
            if prob[idx] > 0.75:
                copyfile(f'{path}/{files[idx]}', f'{dest[p]}/{files[idx]}')