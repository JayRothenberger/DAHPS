# Distributed Asynchronous Hyperparameter Search management utility for SLURM

This module serves as a method of coordinating hyperparameter search runs on a distributed system with access to a reliable shared storage system.  Because it utilizes sqlite3 as a backend it cannot guarantee Availability, but it does guarantee Consistency.  It also guarantees Partition Tolerance as long as the underlying shared storage system is.  In practice the Availability is sufficient for the vast majority of cases as long as the elements of the distributed system are networked such that the write latency is below ~2s.  This rarely is an issue.

This module can be used with non-SLURM allocated systems, but has only been tested with SLURM systems.  In principle the only dependencies of the module are PyTorch, it does not rely on any particular resource management utility.


### Usage Example:

```python 
# config.py

# handles the low resolution datasets
low_res_config = {
    'key': 'image_size',
    'values': [64, 128],
    'default': {
        'key': 'crop_size',
        'values': [16, 32, 64],
        'default': None
    }
}
# handles the high resolution datasets
high_res_config = {
    'key': 'image_size',
    'values': [128, 256],
    'default': {
        'key': 'crop_size',
        'values': [16, 32, 64, 128],
        'default': None
    }
}

dataset_config = {
    'key': 'dataset',
    'values': ['caltech101', 'caltech256', 'food101', 'inat2021', 'cifar10', 'cifar100'],
    'default': high_res_config,
    'cifar10': low_res_config,
    'cifar100': low_res_config
},

optimization_config = {
    'key': 'learning_rate',
    'values': [1e-3, 1e-4],
    'default': {
        'key': 'batch_size',
        'values': [64, 128, 256, 512],
        'default': dataset_config
    }
}

deit_config = {
    'key': 'activation',
    'values': [],
    'default': {
        'key': 'patch_size',
        'values': [1, 2, 4, 16],
        'default': {
            'key': 'num_heads',
            'values': [3, 6, 12],
            'default': {
                'key': 'embed_dim',
                'values': [192, 384, 768],
                'default': {
                    'key': 'conv_first',
                    'values': [True, False],
                    'default': optimization_config
                }
            }
        }
    }
}

resnet_config = {
    'key': 'model',
    'values': ['complex_resnet18', 'complex_resnet34', 'complex_resnet50', 'complex_resnet101'],
    'default': optimization_config
}

model_config = {
    'key': 'model_type',
    'values': ['deit', 'resnet'],
    'default': resnet_config,
    'deit': deit_config
}

preprocessing_config = {
    'key': 'normalize',
    'values': [True, False],
    'default': {
        'key': 'magphase',
        'values': [True, False],
        'default': {
            'key': 'symlog',
            'values': [True, False],
            'default': model_config
        }
    }
}

experiment_config = {
    'root': {
        'key': 'experiment_type',
        'values': ['shearlet', 'fourier', 'baseline'],
        'default': model_config,
        'shearlet': {
            'key': 'n_shearlets',
            'value': [1, 3, 10],
            'default': preprocessing_config
        },
        'fourier': preprocessing_config
    },
    'check_unique': True,
    'repetitions': 1
}

```

```python
import torch
from config import experiment_config as config
from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters


def training_process(args, rank, world_size):

    ...
    
    # return a metric that orders the performance of the models and the model state

    return states, metric


def create_parser():
    parser = argparse.ArgumentParser(description="Shearlet NN")

    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: 10)"
    )
    parser.add_argument(
        "--path", type=str, default='./hp_test', help="path for the hyperparameter search data"
    )


    return parser


def main(args, rank, world_size):
    ...

    device = rank % torch.cuda.device_count()
    print(f'rank {rank} running on device {device} (of {torch.cuda.device_count()})')
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.to_namespace(agent.combination)

    states, metric = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)
        agent.finish_combination(metric)

    ...

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # environment variables from torchrun
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
```