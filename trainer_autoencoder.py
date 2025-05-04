import torch
import numpy as np
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------------------#
# TRAINING AND TESTING
class AutoencoderTrainer:

    def __init__(self, device, optimizer=None, scheduler=None, patience: int = 32, save_loc: str = ""):
        super(AutoencoderTrainer, self).__init__()

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.save_loc = save_loc

    @staticmethod
    def compute_sim_matrix(t_emb, q_emb, dt_func):

        q_dist = dt_func(torch.from_numpy(t_emb), torch.from_numpy(q_emb))

        return np.around(q_dist.squeeze().numpy(), decimals=4)

    # SAVING FUNCTIONS
    def __save_model(self, model, epochs: int, loss: float, check_type: str = ""):

        torch.save({'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss},
                   self.save_loc + check_type)

    @staticmethod
    def __mse_loss(recon_x, x):
        return F.mse_loss(recon_x, x, reduction="sum")

    @staticmethod
    def __kl_divergence_loss(mu, logvar):

        max_logstd = 10
        logstd = logvar.clamp(max=max_logstd)
        kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - (logstd.exp() + 1e-8) ** 2, dim=1))

        # Limit numeric errors
        kl_div = kl_div.clamp(max=100)
        return kl_div

    # train step function
    def _train_step(self, model, loader):

        # set model to training mode
        model.train()

        # training metrics
        total_loss, batch_counter = 0.0, 0

        # iterate over all batches
        for batch_id, batch in enumerate(loader):

            # zeroing out all gradients
            self.optimizer.zero_grad()

            # get models output
            batch, y_true = batch.to(self.device), batch.y.to(self.device)
            x, x_recon, _ = model(batch)

            # get losses
            mse_loss = self.__mse_loss(x_recon, x)
            loss = mse_loss

            # backward prop and optim
            loss.backward()
            self.optimizer.step()

            # update trained metrics
            total_loss += loss.item()

            # increase batch counter
            batch_counter += 1

        epoch_loss = round(total_loss / batch_counter, 4)

        return epoch_loss

    # main training function looping through all epochs
    def train_loop(self, model, tr_loader, num_epochs: int):

        # output container
        loss_stats, early_stop_ctr = [], 0

        # tracking validation loss for checkpoints
        best_val_loss = float('inf')

        # go through all epochs
        try:
            for epoch in range(num_epochs):

                # perform training step
                epoch_tr_loss = self._train_step(model=model, loader=tr_loader)

                # appending
                loss_stats.append(epoch_tr_loss)

                print(f"Epoch [{epoch+1}|{num_epochs}] ==> tr_loss: {epoch_tr_loss:.5f}")

                self.scheduler.step(epoch_tr_loss)

                # check if current validation loss is lower than current best validation loss
                if epoch_tr_loss < best_val_loss:

                    # reset early stopper
                    early_stop_ctr = 0

                    # set new best validation loss
                    best_val_loss = epoch_tr_loss

                    # print event
                    print(f"Epoch [{epoch+1}|{num_epochs}]: new best training loss {best_val_loss:.5f} ==> model checkpoint\n")

                    # save model checkpoint based on decreasing validation loss
                    self.__save_model(model=model, epochs=epoch+1, loss=epoch_tr_loss, check_type=f"_check.pth")

                else:
                    early_stop_ctr += 1
                    print(f"Epoch [{epoch+1}|{num_epochs}| Stopper] @{early_stop_ctr}\n")

                    if early_stop_ctr > self.patience:
                        print(f"Early stopping @epoch {epoch}\n")

                        return loss_stats

        except KeyboardInterrupt:
            print("[KEYBOARD INTERRUPT]: exiting from training because of STRG+C!")

        # save complete model
        self.__save_model(model=model, epochs=num_epochs, loss=epoch_tr_loss, check_type=f"_final.pth")
        print(f"[FIN]: Training complete ==> saved to [{self.save_loc}]...")

        # return metrics
        return loss_stats

