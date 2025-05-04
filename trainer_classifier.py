import torch
import torch.nn as nn
from operator import add
from metrics import MetricTracker
from utils import reshape_labels


# ---------------------------------------------------------------------------------------------------------------------#
# TRAINING AND TESTING
class Trainer:

    def __init__(self, device, criterion, optimizer=None, scheduler=None, patience: int = 32, unfreeze: int = 20,
                 loss_threshold: float = 0.005, model_save: str = ""):
        super(Trainer, self).__init__()

        self.device = device
        self.criterion = criterion
        self.loss_func = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.unfreeze = unfreeze
        self.loss_threshold = loss_threshold
        self.model_save = model_save

        self.predictor = nn.Sigmoid()

        # load metrics module
        self.metric_tracker = MetricTracker(criterion=criterion)

    # SAVING FUNCTIONS
    def __save_model(self, model, epochs: int, loss: float, check_type: str = ""):

        torch.save({'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss},
                   self.model_save + check_type)

    """
    def __reshape_labels(self, yh, yt):

        if isinstance(self.criterion, nn.BCELoss) or isinstance(self.criterion, nn.BCEWithLogitsLoss):
            yh = yh.view(-1)

        return yh, yt.view(-1)
    """

    def __decode_header(self, header, header_lens):

        # all headers within a batch concatenated into one string
        batch_header = "".join(chr(c) for c in header.cpu())

        header = []

        # splitting up individual header per sample in batch
        for i, h_len in enumerate(header_lens):
            header += [batch_header[i:i+h_len]]

        return header

    @staticmethod
    def __update_metrics(batch_metrics, current_metrics):

        assert len(batch_metrics) == len(current_metrics)

        return list(map(add, batch_metrics, current_metrics))

    # train step function
    def train_eval(self, model, loader, is_train: bool = False):

        # set model to training mode
        if is_train:
            model.train()

        else:
            model.eval()

        # training metrics
        batch_counter = 0

        # init metrics
        batch_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]

        # prediction containers
        ytrue_stats, yhat_stats, head_stats = [], [], []

        with torch.set_grad_enabled(is_train):

            # iterate over all batches
            for batch_id, batch in enumerate(loader):

                # zeroing out all gradients
                self.optimizer.zero_grad()

                # cast data onto devices
                batch, y_true = batch.to(self.device), batch.y
                y_hat, y_true = reshape_labels(criterion=self.criterion, yh=model(batch), yt=y_true)

                ytrue_stats.append(y_true)
                yhat_stats.append(y_hat)
                head_stats.append(self.__decode_header(batch.head, batch.head_len))

                # update loss
                y_true_float = y_true.type(torch.float).to(self.device)
                y_hat_float = y_hat.type(torch.float).to(self.device)
                loss = self.criterion(y_hat_float, y_true_float)                        # no predictor needed

                if is_train:

                    # backward propagation
                    loss.backward()
                    self.optimizer.step()

                # handling training metrics
                y_hat_sigmoid = self.predictor(y_hat)
                metrics = self.metric_tracker.compute_metrics(y_true=y_true, y_hat=y_hat_sigmoid)
                metrics.append(loss.item())

                # update running loss and other metrics
                batch_metrics = self.__update_metrics(batch_metrics, metrics)

                # increase batch counter
                batch_counter += 1

            # compute average metrics per epoch
            epoch_metrics = [x / batch_counter for x in batch_metrics]

            return epoch_metrics, [yhat_stats, ytrue_stats, head_stats]

    # main training function looping through all epochs
    def train_loop(self, model, loader, num_epochs: int, k: int):

        # output container
        loss_stats, metric_stats, early_stop_ctr = [], [], 0

        # tracking validation loss for checkpoints
        best_val_loss = float('inf')

        # go through all epochs
        try:
            for epoch in range(num_epochs):

                # unfreeze autoencoder parameters
                if epoch == self.unfreeze:

                    for param in model.parameters():
                        param.requires_grad = True

                    # restart optimizer
                    self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

                # perform training step
                metrics, stats = self.train_eval(model=model, loader=loader, is_train=True)

                ep_tr_loss = metrics[-1]

                # saving stats and metrics per epoch
                loss_stats.append(ep_tr_loss)
                metric_stats.append(stats)

                print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}] ==> tr_loss: {ep_tr_loss:.5f} | tr_F1: {metrics[-2]}")

                self.scheduler.step(metrics[-1])

                if ep_tr_loss < self.loss_threshold:

                    print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}]: loss threshold undercut {ep_tr_loss} < {self.loss_threshold}\n")
                    self.__save_model(model=model, epochs=epoch+1, loss=ep_tr_loss, check_type=f"_check.pth")

                    return loss_stats, metric_stats

                # check if current validation loss is lower than current best validation loss
                if ep_tr_loss < best_val_loss:

                    # reset early stopper
                    early_stop_ctr = 0

                    # set new best validation loss
                    best_val_loss = ep_tr_loss

                    # print event
                    print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}]: new best training loss {best_val_loss:.5f} ==> model checkpoint\n")

                    # save model checkpoint based on decreasing validation loss
                    self.__save_model(model=model, epochs=epoch+1, loss=ep_tr_loss, check_type=f"_check.pth")

                else:
                    early_stop_ctr += 1
                    print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}| Stopper] @{early_stop_ctr}\n")

                    if early_stop_ctr > self.patience:
                        print(f"Early stopping fold 0{k} @epoch {epoch}\n")

                        return loss_stats, metric_stats

            self.__save_model(model=model, epochs=epoch + 1, loss=ep_tr_loss, check_type=f"_final.pth")
            print(f"[FIN]: Training complete ==> saved to [{self.model_save}]...")

            return loss_stats, metric_stats

        except KeyboardInterrupt:
            print("[KEYBOARD INTERRUPT]: exiting from training because of STRG+C!")
