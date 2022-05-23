from pytorch_lightning import LightningModule


# could be improved with some additional metrics for logging:
# train_step_metrics, train_epoch_metrics, val_epoch_metrics
class LitModel(LightningModule):
    def __init__(self, model, optimizer_config, loss_function, val_step_metrics = []):
        super(LitModel, self).__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.loss_function = loss_function
        self.val_step_metrics = val_step_metrics

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer_config = dict(self.optimizer_config)
        self.optimizer_config['lr_scheduler'] = dict(self.optimizer_config['lr_scheduler'])
        self.optimizer_config['optimizer'] = \
            self.optimizer_config['optimizer'](self.model.parameters())
        self.optimizer_config['lr_scheduler']['scheduler'] = \
            self.optimizer_config['lr_scheduler']['scheduler'](self.optimizer_config['optimizer'])
        return self.optimizer_config

    def training_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        metrics_dict = {}
        for metric_dict in self.val_step_metrics:
            metric_key, metric_func = list(metric_dict.items())[0]
            metric_value = metric_func(y_hat, y)
            metrics_dict[metric_key] = metric_value
        self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return metrics_dict

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}