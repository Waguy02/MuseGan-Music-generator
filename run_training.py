import argparse
import logging
import os

import torch.utils.data
from torch.optim import Adam

from constants import EXPERIMENTS_DIR
from logger import setup_logger
from networks.network import CustomNetwork
from training.trainer import Trainer
def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00005, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    return parser.parse_args()

def main(args):
    model_name = "base_model" if args.model_name is None else args.model_name
    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    network=CustomNetwork(experiment_dir=experiment_dir)
    optimizer = Adam(network.parameters(), lr=args.learning_rate)
    loss=None #TODO: Instanciate the loss
    logging.info("Training : "+model_name)
    trainer = Trainer(network,
                      loss,
                      optimizer=optimizer,
                      scheduler=None, ##TODO : Define a custom scheduler,
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )

    train_dataset=None #TODO : Instanciate a train dataset
    val_dataset=None #TODO : Instanciate a val dataset


    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.num_workers)


    trainer.fit(train_dataloader,val_dataloader)
    
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 