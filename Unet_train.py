from torch import optim
from net import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = 'D:/Desktop/learn_tongue/UNET/'
save_path = 'train_image'

transform = transforms.Compose([
    transforms.ToTensor()
])

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0,0,0))
    mask.paste(img, (0,0))
    mask = mask.resize(size)
    return mask


class MyDataset(Dataset):
    def __init__(self, path):  # 拿到标签文件夹中图片的名字
        self.path = path
        self.name = os.listdir(os.path.join(path, 'notedata'))

    def __len__(self):  # 计算标签文件中文件名的数量
        return len(self.name)

    def __getitem__(self, index):  # 将标签文件夹中的文件名在原图文件夹中进行匹配（由于标签是png的格式而原图是jpg所以需要进行一个转化）
        segment_name = self.name[index]  # XX.png
        segment_path = os.path.join(self.path, 'notedata', segment_name)
        image_path = os.path.join(self.path, 'ordata', segment_name.replace('png', 'jpg'))  # png与jpg进行转化

        segment_image = keep_image_size_open(segment_path)  # 等比例缩放
        image = keep_image_size_open(image_path)  # 等比例缩放

        return transform(image), transform(segment_image)


# 设置最大训练轮数
max_epochs = 10
epoch = 1

if __name__ == "__main__":
    dic = []  ###

    data_loader = DataLoader(MyDataset(data_path), batch_size=3, shuffle=True)
    net = UNet().to(device)

    # 加载之前保存的参数
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('Success loading weights')
    else:
        print('No pre-trained weights found')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    while epoch <= max_epochs:
        avg = []  ###

        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print('{}-{}-train_loss===>>{}'.format(epoch, i, train_loss.item()))

            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.jpg')

            avg.append(float(train_loss.item()))  ###

        loss_avg = sum(avg) / len(avg)

        dic.append(loss_avg)

        epoch += 1

    print(dic)
