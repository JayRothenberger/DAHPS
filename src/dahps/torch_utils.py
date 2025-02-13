import random
import time
import os
import pickle

import torch


def sync_parameters(args, rank, search_space, agent_class):
    """
    Synchronizes the parameters between the ranks of a process that is performing
    a single model training run.

    Arguments

    args : argparse.Namespace
        Arguments parsed by an argparse.ArgumentParser.  The arguments parsed which are to
        be used in the search_space should be lists of values which the arguments can take.
        The other arguments should be singletons which will be used by the script.
    rank : int
        Rank of this process in the process world
    search_space : list[str, ...]
        List of keys from the args which the hyperparameter search agent searches over.
        These are the parameters which need to be synchronized between process ranks.
    agent_class : object
        One of the classes in this file (or a custom class that you define) whose parameters
        will be synchronized across the ranks.

    Returns

    hyperparameter search agent class instance
    """
    # the path to the hyperparameter search directory
    path = args.path
    # 1. create the agent
    if rank == 0:
        try:
            agent = agent_class(path, search_space, args)
        except SystemError as e:
            time.sleep(random.randrange(0, 60))
            agent = agent_class(path, search_space, args)
    # 2. generate and broadcast a unique integer - this will specify a path
    # broadcast the integer
    agree = random.randrange(0, 2**32)
    agree = torch.Tensor([agree]).to(rank % torch.cuda.device_count())

    torch.distributed.broadcast(agree, 0)
    torch.distributed.all_reduce(agree)
    agree = int(agree.cpu()[0])

    if rank == 0:
        try:
            os.mkdir(os.path.join(path, f"{agree}"))
        except Exception as e:
            print(e)
        print(path)
        with open(os.path.join(path, f"{agree}/hparams.pkl"), "wb") as fp:
            pickle.dump(agent, fp)
    else:
        time.sleep(20)

    # load the mutual file which holds the hyperparameters
    with open(os.path.join(path, f"{agree}/hparams.pkl"), "rb") as fp:
        agent = pickle.load(fp)
    # return the synchronized agent object
    return agent
