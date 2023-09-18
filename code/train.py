import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from model import get_model
from data import get_data
from loss import get_loss


def train(
    model_name,
    traindata,
    checkpoint_path,
    model_save_path,
    loss_name,
    epochs,
    lr=4e-3,
    weight_decay=0,
    step_size=20,
    gamma=0.5,
    batch_size=16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    loss_f = get_loss(loss_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_last = 99999
    best_model_name = ""
    epochs = epochs
    for epoch in range(1, epochs + 1):
        loss_total = 0
        for step, (inputs, labels) in tqdm(
            enumerate(trainloader),
            desc=f"Epoch {epoch}/{epochs}",
            ascii=True,
            total=len(trainloader),
        ):
            # 原始图片和标签
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = loss_f(out, labels)
            loss_total += loss.item()
            # 梯度清零
            optim.zero_grad()
            # 梯度反向传播
            loss.backward()
            optim.step()
        scheduler.step()
        # 损失小于上一轮则添加
        loss_train = loss_total / len(trainloader)
        if loss_train < loss_last:
            loss_last = loss_train
            torch.save(
                model.state_dict(),
                model_save_path + "model_epoch{}_loss{}.pth".format(epoch, loss_train),
            )
            best_model_name = model_save_path + "model_epoch{}_loss{}.pth".format(
                epoch, loss_train
            )
        print(f"Epoch: {epoch}/{epochs},Loss:{loss_train}")
    print(f"best model is:{best_model_name}")


train(
    model_name="deeplabv3p",
    traindata=get_data("train"),
    checkpoint_path="best_signal_deeplabv3p_resum_hardaug_v2/model_epoch97_loss0.06102542495727539.pth",
    model_save_path="best_signal_deeplabv3p_resum_hardaug_v3/",
    loss_name="dice",
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    step_size=20,
    gamma=0.7,
    batch_size=8,
)
