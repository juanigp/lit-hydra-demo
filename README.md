# lit-hydra-demo
A concept for using PyTorch Lightning + Hydra

## Explanation
The idea is to have a easy way to configure PL's Trainer, Module, and a DataModule (https://www.pytorchlightning.ai/), illustrated with CIFAR10 classification.

### Module:
A PL Module is basically a model plus some code to configure its optimizer, define training and validation step, etc. For a regular classification task, what a Module should do is pretty well established, and we just have some decision choices, like what NN to use, which metric to optimize, the optimizer and its hyperparams, and so on.
The **LitModel** class is supposed to encapsulate this concept: to its constructor we pass a model, a dict with the optimizer configuration, a loss function to optimize, and some metrics to report.

### DataModule:
The concept of PL's DataModule is to contain all the PyTorch DataLoaders for the experiment (training, test, validation and prediction splits). The class **LitDataloadersContainer** implements this idea. To its constructor we pass the DataLoader objects that we will use on the experiment.

### Configurations:
The configuration of the experiment is handled with Hydra configurations (https://mit-ll-responsible-ai.github.io/hydra-zen). We can make a .yml file which contains the fields:
- **trainer**, with the arguments to construct our trainer
- **datamodule**, with the arguments to construct our datamodule
- **module**, with the arguments to construct our module.

Now, for example, our DataModule object requires some DataLoader object, and to construct a DataLoader we need a Dataset object, and for this dataset maybe we want to specify some transformation! All this nested instantiation of objects can be specified in the .yml without writing extra code. For a classification task, we can specify all the objects that we need and their arguments and hyperparameters in our configuration file, and Hydra will handle their instantiation.

### Running:
Currently there is no env, but the basic requirements are PyTorch, PyTorch Lightning and Hydra/hydra-zen.

You can run `$ python3 train.py -cp configs -cn config_cifar.yml`. 

Alternatively you can use **train_classifier_slurm.py** and **slurm_job_demo.sh** to launch the training in a SLURM cluster, automatically using the number of nodes and GPUs that were requested.
