import json
import logging
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import DEVICE
from losses.wassertein_loss import GradientPenalty, WassersteinLoss
from my_utils import Averager
from networks.muse_gan import MuseGan
class Trainer:
    """
    Class to manage the full training pipeline
    """
    def __init__(self, network:MuseGan,
                 g_optimizer,
                 c_optimizer,
                 nb_epochs=10,
                 repeat=5,
                 batch_size=128,
                 reset=False):
        """
        @param network:
        @param dataset_name:
        @param images_dirs:
        @param loss:
        @param optimizer:
        @param nb_epochs:
        @param nb_workers: Number of worker for the dataloader
        """
        self.network = network
        self.batch_size = batch_size
        self.repeat=repeat
        self.g_optimizer=g_optimizer
        self.c_optimizer = c_optimizer
        
        self.g_criterion = WassersteinLoss().to(DEVICE)
        self.c_criterion = WassersteinLoss().to(DEVICE)
        
        
        self.c_penalty=GradientPenalty().to(DEVICE)
        
        self.nb_epochs = nb_epochs
        self.experiment_dir = self.network.experiment_dir
        self.model_info_file = os.path.join(self.experiment_dir, "model.json")
        self.model_info_best_file = os.path.join(self.experiment_dir, "model_best.json")

        if reset:
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.start_epoch = 0
        if not reset and os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                self.start_epoch = json.load(f)["epoch"] + 1
                self.nb_epochs += self.start_epoch
                logging.info("Resuming from epoch {}".format(self.start_epoch))


    def save_model_info(self, infos, best=False):
        json.dump(infos, open(self.model_info_file, 'w'),indent=4)
        if best: json.dump(infos, open(self.model_info_best_file, 'w'),indent=4)

    def fit(self,train_dataloader):
        logging.info("Launch training on {}".format(DEVICE))
        self.network.train()
        self.network.to(DEVICE)
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir)
        itr = self.start_epoch * len(train_dataloader) * self.batch_size  ##Global counter for steps
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                model_info = json.load(f)
                lr=model_info["lr"]
                logging.info(f"Setting lr to {lr}")
                for g in self.optimizer.param_groups:
                    g['lr'] = lr
        if os.path.exists(self.model_info_best_file):
            with open(self.model_info_best_file, "r") as f:
                best_model_info = json.load(f)
                best_loss = best_model_info["val_loss"]

        self.alpha = torch.rand((self.batch_size, 1, 1, 1, 1)).requires_grad_().to(DEVICE)
        for epoch in range(self.start_epoch, self.nb_epochs):  # Training loop
            epoch_gloss = Averager()
            epoch_cfloss = Averager()
            epoch_crloss = Averager()
            epoch_cploss = Averager()
            epoch_closs = Averager()
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.nb_epochs}")
            for _, real in enumerate(pbar):
                """
                Training lopp
                """
                itr+=1
                #Train the critic
                real=real.to(DEVICE)
                batch_closs = Averager()
                batch_cfloss = Averager()
                batch_crloss = Averager()
                batch_cploss = Averager()
                for _ in range(self.repeat):
                    cords = torch.randn(self.batch_size, 32).to(DEVICE)
                    style = torch.randn(self.batch_size, 32).to(DEVICE)
                    melody = torch.randn(self.batch_size, 4, 32).to(DEVICE)
                    groove = torch.randn(self.batch_size, 4, 32).to(DEVICE)

                    self.c_optimizer.zero_grad()
                    with torch.no_grad():
                        fake = self.network.generator(cords, style, melody, groove).detach()
                    realfake = self.alpha * real + (1. - self.alpha) * fake

                    fake_pred = self.network.critic(fake)
                    real_pred = self.network.critic(real)
                    realfake_pred = self.network.critic(realfake)
                    fake_loss = self.c_criterion(fake_pred, - torch.ones_like(fake_pred))
                    real_loss = self.c_criterion(real_pred, torch.ones_like(real_pred))
                    penalty = self.c_penalty(realfake, realfake_pred)
                    closs = fake_loss + real_loss + 10 * penalty
                    closs.backward(retain_graph=True)
                    self.c_optimizer.step()
                    batch_cfloss.send(fake_loss.item())
                    batch_crloss.send(real_loss.item())
                    batch_cploss.send(10 * penalty.item())
                    batch_closs.send(closs.item() / self.repeat)


                # Train Generator
                self.g_optimizer.zero_grad()
                cords = torch.randn(self.batch_size, 32).to(DEVICE)
                style = torch.randn(self.batch_size, 32).to(DEVICE)
                melody = torch.randn(self.batch_size, 4, 32).to(DEVICE)
                groove = torch.randn(self.batch_size, 4, 32).to(DEVICE)

                fake = self.network.generator(cords, style, melody, groove)
                fake_pred = self.network.critic(fake)
                b_gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                b_gloss.backward()
                self.g_optimizer.step()

                """
                4.Writing logs and tensorboard data, loss and other metrics
                """
                batch_data={
                "generator_loss":b_gloss.item(),
                "critic_loss":batch_closs.value,
                "critic_fake_loss":batch_cfloss.value,
                "critic_real_loss":batch_crloss.value,
                "critic_penalized_loss":batch_cploss.value
                    }

                for k,v in batch_data.items():
                    self.summary_writer.add_scalar(f"Train steps/{k}", v, itr)

                epoch_gloss.send(b_gloss.item())
                epoch_cfloss.send(batch_cfloss.value)
                epoch_crloss.send(batch_crloss.value)
                epoch_cploss.send(batch_cploss.value)
                epoch_closs.send(batch_closs.value)



            epoch_data={
                "generator_loss":epoch_gloss.value,
                "critic_loss":epoch_closs.value,
                "critic_fake_loss":epoch_cfloss.value,
                "critic_real_loss":epoch_crloss.value,
                "critic_penalized_loss":epoch_cploss.value
            }
            for k,v in epoch_data.items():
                self.summary_writer.add_scalar(f"Train epochs/{k}",v,epoch)
            logging.info(f"Epoch {epoch}/{self.nb_epochs} | Generator loss: {epoch_gloss.value:.3f} | Critic loss: {epoch_closs.value:.3f}")
            logging.info(f"Critic performance (fake: {epoch_cfloss.value:.3f}, real: {epoch_crloss.value:.3f}, penalty: {epoch_cploss.value:.3f})")

            #TODO write epoch metrics results
            infos = epoch_data
            infos["epoch"]=epoch
            infos["batch_size"]=self.batch_size
            infos["generator_lr"]=self.g_optimizer.param_groups[0]['lr']
            infos["critic_lr"] = self.c_optimizer.param_groups[0]['lr']
            self.network.save_state()
            self.save_model_info(infos)
            

    


