import pytorch_lightning as pl
import hydra
from hydra_zen import instantiate 

@hydra.main(config_path=None)
def task_function(cfg):
    pl.utilities.seed.seed_everything(42)
    obj = instantiate(cfg)
    module = obj.module
    datamodule = obj.datamodule
    trainer = obj.trainer()
    trainer.fit(module, datamodule=datamodule)
    return 0

if __name__ == '__main__':
    task_function()
