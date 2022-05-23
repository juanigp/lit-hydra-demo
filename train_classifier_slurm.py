import os
import pytorch_lightning as pl
import hydra
from hydra_zen import instantiate 

# SLURM env variables
# LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
# GLOBAL_RANK = int(os.environ['SLURM_PROCID'])
WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
STRATEGY = None if WORLD_SIZE == 1 else 'ddp'
NUM_NODES = int(os.environ['SLURM_JOB_NUM_NODES'])
TASKS_PER_NODE = int(os.environ['SLURM_TASKS_PER_NODE'][0])

@hydra.main(config_path=None)
def task_function(cfg):
    pl.utilities.seed.seed_everything(42)
    obj = instantiate(cfg)
    module = obj.module
    datamodule = obj.datamodule
    trainer = obj.trainer(    
        strategy=STRATEGY,
        gpus=TASKS_PER_NODE,
        num_nodes=NUM_NODES,)

    trainer.fit(module, datamodule=datamodule)
    return 0

if __name__ == '__main__':
    task_function()