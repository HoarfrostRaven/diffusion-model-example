from transformers import PretrainedConfig


class UnetConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(
        self,
        input_channels: int = 3,
        num_features: int = 256,
        num_context_features: int = 10,
        width: int = 28,
        height: int = 28,
        **kwargs,
    ):
        self.input_channels = input_channels
        self.num_features = num_features
        self.num_context_features = num_context_features
        self.width = width
        self.height = height
        super().__init__(**kwargs)


# unet_config = UnetConfig()
# unet_config.save_pretrained("base-unet")
