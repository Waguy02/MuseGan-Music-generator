import argparse
import json
import logging
import os

import torch.utils.data
from torch.optim import Adam

from constants import EXPERIMENTS_DIR, DATASET_FILE
from dataset.dataset import LPDDataset, MidiDataset
from logger import setup_logger
from my_utils import read_json
from networks.muse_gan import MuseGan
from training.trainer import Trainer
def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate of Adam optimized")
    parser.add_argument("--nb_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")

    parser.add_argument("--z_dimension", type=int, default=32, help="Z(noise)-space dimension.")
    parser.add_argument("--g_channels", type=int, default=1024, help="Generator hidden channels.")
    parser.add_argument("--g_features", type=int, default=1024, help="Generator hidden features.")
    parser.add_argument("--g_lr", type=float, default=0.001, help="Generator learning rate.")
    parser.add_argument("--c_channels", type=int, default=128, help="Critic hidden channels.")
    parser.add_argument("--c_features", type=int, default=1024, help="Critic hidden features.")
    parser.add_argument("--c_lr", type=float, default=0.001, help="Critic learning rate.")
    parser.add_argument("--n_bars",type=int,default=5,help="Number of bars per sample")
    parser.add_argument("--step_bars", type=int, default=48, help="Number of step per bar")
    parser.add_argument("--n_tracks", type=int, default=4, help="Number of tracks per sample")
    parser.add_argument("--n_pitches",type=int,default=84,help="Number of pitches considered")



    parser.add_argument("--log_level", "-l", type=str, default="INFO")



    return parser.parse_args()

def main(args):
    model_name = "base_model" if args.model_name is None else args.model_name
    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    config_file=os.path.join(experiment_dir,"config.json")

    if not os.path.exists(experiment_dir):os.makedirs(experiment_dir)
    if  not os.path.exists(config_file) or args.reset:
        model_config={
        "z_dimension":args.z_dimension,
        "g_channels": args.g_channels,
        "g_features" : args.g_features,
        "c_channels" : args.c_channels,
        "c_features" : args.c_features,
        "n_bars":args.n_bars,
        "step_bars":args.step_bars,
        "n_tracks":args.n_tracks,
        "n_pitches":args.n_pitches
        }
        with open(config_file,"w") as f:json.dump(model_config,f,indent=4)



    network=MuseGan(experiment_dir=experiment_dir,
                    reset=args.reset,
                    )


    g_optimizer = Adam(network.generator.parameters(),lr=args.g_lr,betas=(0.5, 0.9))
    c_optimizer = Adam(network.critic.parameters(),lr=args.c_lr,betas=(0.5, 0.9))

    logging.info("Training : "+model_name)
    trainer = Trainer(network,
                      g_optimizer,
                      c_optimizer,
                      nb_epochs= args.nb_epochs,
                      batch_size=args.batch_size,
                      reset=args.reset,
                      )

    train_dataset=LPDDataset(DATASET_FILE)
    # train_dataset = MidiDataset(DATASET_FILE,split="nonzero")
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True,drop_last=True)
    trainer.fit(train_dataloader)

    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 