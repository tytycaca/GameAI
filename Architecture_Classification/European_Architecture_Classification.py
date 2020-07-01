from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid

# CNN 정의.
# 기존 Cifar10 tutorial 을 개조한 모델에서는 수정을 거듭해도 정확도가 상승하지 않아, 새롭게 AlexNet 모델을 참고하여 정의함.
# 원본을 간소화 하였음. (convolutional, maxpooling, dropout 등의 layer 개수 하향조정)
# 혼동을 방지하기 위해 파라매터 표기.
# Cifar10 의 리사이징된 32x32 사이즈 이미지와 달리 128x128 이미지를 100x100 사이즈로 크롭하여 사용. (AlextNet 은 224x224 사용)
# convolutional layer 추가. (AlexNet 참조)
# convolutional layer, LRN layer 추가 및 입력층 사이즈 변화 등으로 인한 output size 변화에 맞게 full connected layer 의 입력층을 32 * 4 * 4 로 수정.
# flatten layer 또한 맞게 수정. (32 * 4 * 4)
# ReLU 함수의 단점인 양수 방향으로 무한하게 값이 커지는 것을 방지하기 위해 정규화 과정인 LRN Layer 를 추가. (Local Response Normalization)
# over-fitting 을 막기 위한 dropout layer 추가.
# 원본인 AlexNet 과 달리 dropout layer 는 한 번만 넣었음. (over-fitting 의 위험성보다 학습 속도에 좀 더 집중함)
# epoch 1000 기준, 기존 Cifar10 모델에서의 정확도 25% 에서 현재 모델 변경 후 35% 까지 향상됨.
class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
        self.LRN = nn.LocalResponseNorm(3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=384 * 3 * 3, out_features=864)   # 출력층까지 1/4 씩 변화하게 설정, in_features 사이즈 수정
        self.fc2 = nn.Linear(in_features=864, out_features=216)   # 출력층까지 1/4 씩 변화하게 설정
        self.fc3 = nn.Linear(in_features=216, out_features=4)   # 출력층 (Romanesque, Gothic, Renaissance, Baroque 4 가지)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.LRN(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.LRN(x)
        x = self.pool(x)
        # print(x.shape)    # output size 확인용
        x = x.view(-1, 384 * 3 * 3)    # flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
# 이미지를 보여주기 위한 함수.
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
'''

# -------------------------- 전역 설정 ------------------------------------
# 데이터셋 및 저장 경로 정의.
TRAIN_PATH = "./data/Architecture_dataset/train"
TEST_PATH = "./data/Architecture_dataset/test"
SAVE_PATH = './Architecture_net.pth'

# CUDA 장치 설정.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# epoch 횟수와 batch_size 설정.
# data 의 개수가 얼마 안되기에 batch_size 를 크게 잡음. (batch_size=50).
epoch_total = 1000
batch = 50
# -------------------------------------------------------------------------

def main():
    '''
    # 학습용 이미지를 무작위로 가져오기.
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # 이미지 보여주기.
    imshow(make_grid(images, nrow = 10))
    # 정답(label) 출력.
    print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
    '''

    # 데이터 전처리.
    # 정규화를 위해 128 x 128 사이즈로 리사이징.
    # 건축물 사진의 대부분이 가장자리가 하늘이나 구름 등 건물과는 상관없는 요소들이기 때문에 Center Crop 을 사용하여 100 x 100 사이즈로 크롭하여 샘플링함.
    trans = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.CenterCrop(100),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    # 이미지 불러오기.
    trainset = ImageFolder(root=TRAIN_PATH, transform=trans)
    testset = ImageFolder(root=TEST_PATH, transform=trans)

    # Loader 정의.
    # 여기에서 데이터 라벨링 작업도 같이 함.
    # num_workers 값은 일반적으로 코어 개수의 절반정도 수치면 무난하게 시스템 리소스를 사용하여 학습이 가능하다고 하여 3 으로 설정.
    # data 의 개수가 얼마 안되기에 batch_size 를 크게 잡음. (batch_size=50).
    # 속도 증가를 위해 pin_memory 사용.
    trainloader = DataLoader(trainset, batch_size=batch, num_workers=3, pin_memory=True, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch, num_workers=3, shuffle=False)

    # 라벨 분류를 위한 작업.
    classes = ('Romanesque', 'Gothic', 'Renaissance', 'Baroque')

    # 네트워크 선택.
    net = My_Net()

    # 네트워크를 CUDA 장치로 보내기.
    net.to(device)

    # Loss Function 과 Optimizer 정의. (교차 엔트로피 손실(Cross-Entropy loss)과 모멘텀(momentum) 값을 갖는 SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epoch_total):  # 데이터셋을 수차례 반복. (epoch 10 회)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력.
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

    # 학습한 데이터 저장. (PATH 는 코드 상단부에서 일괄로 정의)
    torch.save(net.state_dict(), SAVE_PATH)

    # ---------------------------- 시험용 데이터로 학습 결과 검사 ------------------------------
    # 테스트 이미지셋 불러오기.
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    '''
    # 테스트 이미지셋 출력.
    plt.imshow(make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''

    # 저장된 학습 데이터 불러오기.
    net = My_Net()
    net.load_state_dict(torch.load(SAVE_PATH))

    # 신경망 예측 결과 출력.(부분 출력)
    outputs = net(images)
    outputs
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(10)))

    # 전체 테스트 데이터셋에 대한 정확도.
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 1000 test images: %d %%' % (
            100 * correct / total))

    # 라벨별 정확도.
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(4):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    main()


