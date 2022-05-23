import pytorch_lightning as pl

class LitDataloadersContainer(pl.LightningDataModule):
    def __init__(self, 
        train_dataloader = None,
        val_dataloader = None,
        test_dataloader = None,
        predict_dataloader = None,
    ):
        super().__init__()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader
        self.predict_dataloader_ = predict_dataloader

    def prepare_data(self):
        return

    def setup(self, stage = None):
        return

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_

    def test_dataloader(self):
        return self.test_dataloader_

    def predict_dataloader(self):
        return self.predict_dataloader_