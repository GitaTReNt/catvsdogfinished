import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models
from torch.autograd import Variable
import time
from tqdm import tqdm

#超参数的设置

epochs = 10
batch_size = 16
learning_rate = 0.0001  # 防止忽略细节
#cuda的设置 有则用gpu，没有就选择cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#model = torchvision.models.resnet18(pretrained=False)   # resnet18
model = torch.load('model.pth')


#  ...
#  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
# (fc): Linear(in_features=512, out_features=1000, bias=True) 将features作为输入
#)


nums_in_features = model.fc.in_features # 使用模型的输入特征数
model.fc = nn.Linear(nums_in_features, 2)   #因为是猫狗二分类，所以只需要有两个输出即可
model.to(device)    #模型加载到设备上训练

#-------数据路径-----------------------------------------------------

img_valid_pth = "data/val"
imgs_train_pth = "data/train"

#-------------------数据预处理----------------------------------------
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1)),  # 随机裁剪+resize
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ToTensor(),  # rgb归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用论文的参数
])
#transform_test = transforms.Compose([   # 验证集不再做图像增强
#    transforms.Resize((128, 128)),
 #   transforms.ToTensor(),
#    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#])
#-------------------------导入数据------------------------------
dataset_to_train = datasets.ImageFolder(imgs_train_pth, transform)
#dataset_to_valid = datasets.ImageFolder(img_valid_pth, transform_test) 欠佳，影响acc
dataset_to_valid = datasets.ImageFolder(img_valid_pth, transform)
print(dataset_to_train.class_to_idx)    # 打印 labels
print(dataset_to_valid.class_to_idx)

train_loader = torch.utils.data.DataLoader(dataset_to_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_to_valid, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用adam优化器

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)  # 借用一种优化器对学习率进行优化

#-------------------------训练模式--------------------
def train(epoch):
   # vis =True
   # i = 1
    loss_total = 0
    correct = 0
    #total = len(train_loader.dataset)
    model.train() # 进行训练
    #loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):   # 开始迭代
        data = Variable(data).to(device)
        target = Variable(target).to(device)
        output = model(data)
        optimizer.zero_grad()
        _,predict_label = torch.max(output.data, 1)
        loss = criterion(output, target)
        loss.backward()

        correct += torch.sum(predict_label == target)
        optimizer.step()
        loss_total += loss.data.item()
        # if (batch_idx + 1) % 5 == 0:
        #    print('Train : Epoch =  {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
        #               100. * (batch_idx + 1) / len(train_loader), loss.item()))
        # losses =  loss_total / len(train_loader)
        # i += 1
        # loop.set_description(f'Epoch [{epoch}/{epochs}]')
        # loop.set_postfix(loss=loss.item() / (batch_idx + 1), acc=correct / total)
        # print('\n', "train :  epoch =", epoch,
        # " learn rate =" , optimizer.param_groups[0]['lr'],
        # " loss =", losses /(batch_idx + 1) , " accuracy =", correct / total, '\n')
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
        average_loss = loss_total / len(train_loader)
        print('epoch:{},loss:{}'.format(epoch, average_loss))




def valid(model,valid_loader):
    loss_total = 0
    correct = 0
    total = len(valid_loader.dataset)
    model.eval()
    print(total, len(valid_loader))
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            loss_total += print_loss
        correct = correct.data.item()
        accuracy = correct / total
        losses = loss_total / len(valid_loader)
        scheduler.step(print_loss / epoch + 1)
        print('\nvalidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            losses, correct, len(valid_loader.dataset), 100 * accuracy))

for epoch in range(1, epochs + 1):
    train(epoch)
    valid(model, valid_loader)
torch.save(model, 'model.pth')
