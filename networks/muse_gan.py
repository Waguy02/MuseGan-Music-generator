import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, DEVICE
from losses.wassertein_loss import WassersteinLoss, GradientPenalty
from my_utils import read_json
from networks.discriminator import MuseCritic
from networks.generator import MuseGenerator
from networks.utils import initialize_weights



class MuseGan(nn.Module):
    def __init__(self, experiment_dir="base_muse_gan",
                 reset=False, load_best=True):
        super(MuseGan, self).__init__()
        self.experiment_dir = experiment_dir
        self.model_name = os.path.basename(self.experiment_dir)
        self.reset = reset
        self.load_best = load_best
        self.setup_dirs()





        self.setup_network()

        if not reset: self.load_state()

    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """

        #1: load model config
        config_file=os.path.join(self.experiment_dir,"config.json")
        assert os.path.exists(config_file),f"No config.json found in {self.experiment_dir}"
        model_config=read_json(config_file)
        self.z_dimension =model_config["z_dimension"]
        self.g_channels =model_config["g_channels"]
        self.g_features =model_config["g_features"]
        self.c_channels =model_config["c_channels"]
        self.c_features = model_config["c_features"]
        self.n_bars=model_config["n_bars"]
        self.step_bars=model_config["step_bars"]
        self.n_pitches=model_config["n_pitches"]
        self.n_tracks=model_config["n_tracks"]


        self.generator = MuseGenerator(
            z_dimension=self.z_dimension,
            hid_channels=self.g_channels,
            hid_features=self.g_features,
            out_channels=1,
            n_tracks=self.n_tracks,
            n_pitches=self.n_pitches,
            n_bars=self.n_bars,
            n_steps_per_bar=self.step_bars
        ).to(DEVICE)
        self.generator=self.generator.apply(initialize_weights)


        self.critic=MuseCritic(
            hid_channels=self.c_channels,
            hid_features=self.c_features,
            out_features=1,
            n_tracks=self.n_tracks,
            n_bars=self.n_bars,
            n_steps_per_bar=self.step_bars,
            n_pitches=self.n_pitches
            )
        self.critic=self.critic.apply(initialize_weights)


    ##2. Model Saving/Loading
    def load_state(self, best=False):
        """
        Load model
        :param self:
        :return:
        """
        if best and os.path.exists(self.save_best_file):
            logging.info(f"Loading best model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))
            return

        if os.path.exists(self.save_file):
            logging.info(f"Loading model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))

    def save_state(self, best=False):
        if best:
            logging.info("Saving best model")
            torch.save(self.state_dict(), self.save_best_file)
        torch.save(self.state_dict(), self.save_file)

    ##3. Setupping directories for weights /logs ... etc
    def setup_dirs(self):
        """
        Checking and creating directories for weights storage
        @return:
        """
        self.save_file = os.path.join(self.experiment_dir, f"{self.model_name}.pt")
        self.save_best_file = os.path.join(self.experiment_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)







