import torch

class BaseSampler():
    def __init__(self):
        self.possible_modules = (
            'text_enc_1',
            'text_enc_2',
            'text_enc_3',
            'transformer',
            'vae'
        )

    def to(self, device: torch.device):
        for module_name in self.possible_modules:
            if hasattr(self, module_name):
                getattr(self, module_name).to(device)

        return self

    def remove_attr(self, attr: str):
        if hasattr(self, attr):
            delattr(self, attr)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")