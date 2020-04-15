import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class Module(pl.LightningModule):

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
    ):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss = self.model.loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-3)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss(sr, hr)
        return {
            "loss": loss,
            "mse": metrics.mse(sr, hr),
            "psnr": metrics.psnr(sr, hr)
            }

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self.forward(lr)
        loss = self.loss(sr, hr)
        return {
            "val_loss": loss,
            "val_mse": metrics.mse(sr, hr),
            "val_psnr": metrics.psnr(sr, hr)
            }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_mse': avg_mse,
            'val_psnr': avg_psnr
            }
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    scale_factor = 2
    train_dataset = BSDS300(scale_factor=scale_factor, data_dir="../.data/")
    val_dataset = Set14(scale_factor=scale_factor, data_dir="../.data/")
    model = SRCNN(scale_factor)
    module = Module(
        model,
        train_dataset,
        val_dataset,
        test_dataset
    )
    trainer = pl.Trainer(
        max_nb_epochs=2,
        gpus=1
    )
    trainer.fit(module)
    trainer.test()