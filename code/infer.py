import os
import torch
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import get_model
from data import get_data


# model_name = "deeplabv3plus/model_epoch13_loss0.050909265995025633.pth"  # best
# model_name = "deeplabv3plus/model_epoch19_loss0.04801582956314087.pth"
# model_name = "unet++/model_epoch32_loss0.049111151218414306.pth"  # best
# model_name = "deeplabv3plus/model_epoch16_loss0.04915279150009155.pth"
# model_name = "unet++/model_epoch40_loss0.04605033540725708.pth"
# model_name = "unet++/model_epoch33_loss0.048330595016479495.pth"
# model_name = "unet++_imgaug/model_epoch40_loss0.06442743301391601.pth"
# model_name = "unet++_imgaug/model_epoch46_loss0.06343710851669311.pth"  # 还没有过拟合
model_name = "unet++_imgaug/model_epoch50_loss0.06234789276123047.pth"  # best
# model_name = "unet++_imgaug/model_epoch57_loss0.06153344440460205.pth"  # 过拟合
# model_name = "unet++_imgaug/model_epoch62_loss0.06091019487380982.pth"  # 过拟合
# model_name = "unet++_imgaug/model_epoch91_loss0.06015445613861084.pth"  # 过拟合
model_name = "deeplabv3plus_imgaug/model_epoch89_loss0.06424852800369263.pth"
model_name = "deeplabv3plus_imgaug/_model_epoch95_loss0.06320640420913697.pth"  # best
model_name = "deeplabv3plus_imgaug/_model_epoch70_loss0.06401363897323609.pth"  # 效果很差
model_name = "unet++_imgaug/_model_epoch71_loss0.061881133556365965.pth"
model_name = "deeplabv3plus_imgaug/_model_epoch90_loss0.0634520263671875.pth"
model_name = "deeplabv3plus_imgaug_easy/_model_epoch35_loss0.05353249168395996.pth"
# model_name = "deeplabv3plus_imgaug_easy/_model_epoch47_loss0.050852485656738285.pth"
# model_name = "deeplabv3plus_imgaug_easy/_model_epoch53_loss0.05017601585388184.pth"
# model_name = "deeplabv3plus_imgaug_easy/_model_epoch30_loss0.054397048473358155.pth"
# model_name = "deeplabv3plus_imgaug_easy/_model_epoch40_loss0.05300587701797485.pth"
# 尝试deeplabv3plus_imgaug_easy_bce_dice/model_epoch98_loss0.04264263886213303.pth,


def zip_files(file_paths, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file)


## 普通推理
@torch.no_grad()
def infer(model_name, checkpoint_path, img_save_path, zip_out_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    testdata = get_data("test")
    for i, inputs in tqdm(enumerate(testdata)):
        inputs = inputs.reshape(1, 3, 320, 640).to(device)
        out = model(inputs)
        threshold = threshold
        out = torch.where(
            out >= threshold, torch.tensor(255, dtype=torch.float).to(device), out
        )
        out = torch.where(
            out < threshold, torch.tensor(0, dtype=torch.float).to(device), out
        )
        out = out.detach().cpu().numpy().reshape(1, 320, 640)
        img = Image.fromarray(out[0].astype(np.uint8))
        img = img.convert("1")
        img.save(img_save_path + testdata.name[i])

    # 打包图片
    file_paths = [
        img_save_path + i for i in os.listdir(img_save_path) if i[-3:] == "png"
    ]
    zip_files(file_paths, zip_out_path)


# 带tta的推理
@torch.no_grad()
def infer_with_tta(
    model_name, checkpoint_path, img_save_path, zip_out_path, threshold=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    testdata = get_data("test")
    for i, inputs in tqdm(enumerate(testdata)):
        inputs = inputs.reshape(1, 3, 320, 640).to(device)
        inputs1 = inputs.flip(dims=[2]).to(device)
        inputs2 = inputs.flip(dims=[3]).to(device)
        inputs3 = inputs.flip(dims=[2, 3]).to(device)
        out = model(inputs)
        out1 = model(inputs1).flip(dims=[2])
        out2 = model(inputs2).flip(dims=[3])
        out3 = model(inputs3).flip(dims=[2, 3])
        out = (out + out1 + out2 + out3) / 4
        threshold = threshold
        out = torch.where(
            out >= threshold, torch.tensor(255, dtype=torch.float).to(device), out
        )
        out = torch.where(
            out < threshold, torch.tensor(0, dtype=torch.float).to(device), out
        )
        out = out.detach().cpu().numpy().reshape(1, 320, 640)
        img = Image.fromarray(out[0].astype(np.uint8))
        img = img.convert("1")
        img.save(img_save_path + testdata.name[i])

    file_paths = [
        img_save_path + i for i in os.listdir(img_save_path) if i[-3:] == "png"
    ]
    zip_files(file_paths, zip_out_path)


# infer(
#     model_name="deeplabv3p",
#     checkpoint_path="",
#     img_save_path="infers/",
#     zip_out_path="infers.zip",
# )
infer_with_tta(
    model_name="deeplabv3p",
    checkpoint_path="best_signal_deeplabv3p_resum_v2/model_epoch27_loss0.05051758503913879.pth",
    img_save_path="infers/",
    zip_out_path="infers.zip",
    threshold=0.5,
)

# 1_fold_unet
# epoch 9  0.9491
# epoch 8  0.9513
# best_signal_deeplabv3p
# best dice loss 0.535
# epoch 32 0.9506
# epoch 35 0.9515
# epoch 38 0.9504
# epoch 40 0.9510
# deeplabv3p_se_resnext
# epoch 53 0.9495
# epoch 60 0.9489
# esay_last 0.9483 epoch27 0.9505
# hard_last 0.9484
