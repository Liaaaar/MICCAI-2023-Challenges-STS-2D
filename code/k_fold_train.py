import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import get_model
from data import get_data
from loss import get_loss


@torch.no_grad()
def val(val_loader, model, device, loss_fun):
    model.eval()
    val_loss_total = 0
    for step, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(inputs)
        loss = loss_fun(out, labels)
        val_loss_total += loss.item()
    loss_val = val_loss_total / len(val_loader)
    return loss_val


def train(
    model_name,
    traindataset,
    valdataset,
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
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valdataset = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    loss_f = get_loss(loss_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_last = [99999, 99999]
    best_model_name = ""
    for epoch in range(1, epochs + 1):
        train_loss_total = 0
        for step, (inputs, labels) in enumerate(trainloader):
            # 原始图片和标签
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = loss_f(out, labels)
            train_loss_total += loss.item()
            # 梯度清零
            optim.zero_grad()
            # 梯度反向传播
            loss.backward()
            optim.step()
        scheduler.step()
        loss_train = train_loss_total / len(trainloader)
        loss_val = val(valdataset, model, device, loss_f)
        # 损失小于上一轮则添加
        if loss_val < loss_last[0]:
            loss_last[0], loss_last[1] = loss_val, loss_train
            torch.save(
                model.state_dict(),
                model_save_path
                + "epoch{}_valloss{:.5f}_trainloss{:.5f}.pth".format(
                    epoch, loss_val, loss_train
                ),
            )
            best_model_name = (
                model_save_path
                + "epoch{}_valloss{:.5f}_trainloss{:.5f}.pth".format(
                    epoch, loss_val, loss_train
                )
            )
        print(
            f"Epoch: {epoch}/{epochs},train_Loss:{loss_train:.5f},val_loss:{loss_val:.5f}"
        )
    print(f"best model is:{best_model_name}")


def k_fold_train(
    fold_num,
    model_name,
    checkpoint_path,
    model_save_path,
    loss_name,
    epochs=100,
    lr=1e-3,
    weight_decay=0,
    step_size=20,
    gamma=0.5,
    batch_size=16,
):
    skf = KFold(n_splits=fold_num, shuffle=True)
    dataset = get_data("train")
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(dataset)):
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        temp_save_path = model_save_path + f"fold{fold_idx}/"
        if not os.path.exists(temp_save_path):
            os.makedirs(temp_save_path)
        print(f"training fold {fold_idx}......")
        print(f"checkpoint is saving to {temp_save_path}")
        train(
            model_name=model_name,
            traindataset=train_dataset,
            valdataset=valid_dataset,
            checkpoint_path=checkpoint_path,
            model_save_path=temp_save_path,
            loss_name=loss_name,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            batch_size=batch_size,
        )


k_fold_train(
    fold_num=5,
    checkpoint_path=None,
    model_name="deeplabv3p",
    model_save_path="5_fold_deeplabv3p_with_se_resnext_bcedice/",
    loss_name="dice",
    epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    step_size=20,
    gamma=0.5,
    batch_size=8,
)
# checkpoint_path="deeplabv3plus_imgaug_easy/_model_epoch35_loss0.05353249168395996.pth"
