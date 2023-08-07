from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def load_dataset(is_train):
    transform  = transforms.Compose([transforms.Resize(224),
                                    #transforms.RandomCrop(32, padding=2,padding_mode='reflect'),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),])
    if is_train:
        return ImageFolder('./train', transform=transform)
    else:
        return ImageFolder('./test', transform=transform)