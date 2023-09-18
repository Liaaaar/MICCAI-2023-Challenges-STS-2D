import torch


# dice_loss
def dice_loss(prob, target):
    smooth = 1.0
    # prob = torch.sigmoid(logits)
    batch = prob.size(0)
    prob = prob.view(batch, 1, -1)
    target = target.view(batch, 1, -1)
    intersection = torch.sum(prob * target, dim=2)
    denominator = torch.sum(prob, dim=2) + torch.sum(target, dim=2)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1.0 - dice
    return dice_loss


# bce_loss
def bce_loss():
    return torch.nn.BCELoss()


# bce_dice_loss
def bce_dice_loss(prob, target):
    bce = torch.nn.BCELoss()
    dice = dice_loss
    alpha = 0.2
    return alpha * bce(prob, target) + (1 - alpha) * dice(prob, target)


def get_loss(type):
    if type == "dice":
        return dice_loss
    elif type == "bce":
        return torch.nn.BCELoss()
    elif type == "bce_dice":
        return bce_dice_loss


# loss = get_loss("bce_dice")
# print(loss(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.8, 0.2, 0.3])))
