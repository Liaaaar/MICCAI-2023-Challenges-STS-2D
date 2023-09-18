# 加载一些基础的库
import os
import cv2
import albumentations as A
import torchvision.transforms as T
from torch.utils.data import Dataset


totensor = T.Compose(
    [
        T.ToTensor(),
        # T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)
# totensor = T.ToTensor()
# 具体含义: https://blog.csdn.net/qq_27039891/article/details/100795846
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.RandomBrightnessContrast(p=1),
                A.Blur(p=1),
                A.OpticalDistortion(p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.GridDistortion(p=1),
                A.MotionBlur(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
)


class MyDataset(Dataset):
    def __init__(self, path):
        self.mode = "train" if "mask" in os.listdir(path) else "test"  # 表示训练模式
        self.path = path  # 图片路径
        dirlist = os.listdir(path + "image/")  # 图片的名称
        self.name = [n for n in dirlist if n[-3:] == "png"]  # 只读取图片

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):  # 获取数据的处理方式
        name = self.name[index]
        # 读取原始图片和标签
        if self.mode == "train":  # 训练模式
            ori_img = cv2.imread(self.path + "image/" + name)  # 原始图片
            lb_img = cv2.imread(self.path + "mask/" + name)  # 标签图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            lb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2GRAY)  # 掩膜转为灰度图
            transformed = transform(image=ori_img, mask=lb_img)
            return totensor(transformed["image"]), totensor(transformed["mask"])

        if self.mode == "test":  # 测试模式
            ori_img = cv2.imread(self.path + "image/" + name)  # 原始图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            return totensor(ori_img)


def get_data(
    mode,
    train_path="train/",
    test_path="/home/19170100004/code/MICCAI_2023_Challenges_2D/code/baseline_v2/test/",
):
    if mode == "train":
        return MyDataset(train_path)
    if mode == "test":
        return MyDataset(test_path)


# a = get_data("train")[5]
# print(type(a[0]), type(a[1]), a[0].shape, a[1].shape)
