import segmentation_models_pytorch as smp


def get_model(
    name,
    encoder_name="resnet50",  # efficientnet-b5,se_resnext50_32x4d
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
):
    if name == "deeplabv3p":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif name == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
