import torch
from utils import move_batch_to_device

# ---------------------------------------------------------------------------------------------------------------------#
# TRAINING AND TESTING
class Trainer:

    def __init__(self, device, criterion, optimizer=None, criterion_optim=None, scheduler=None, miner=None,
                 patience: int = 32, unfreeze: int = 20, loss_threshold: float = 0.005, model_save: str = ""):
        super(Trainer, self).__init__()

        self.device = device
        self.loss_func = criterion
        self.loss_optimizer = criterion_optim
        self.miner = miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.unfreeze = unfreeze
        self.loss_threshold = loss_threshold
        self.model_save = model_save

    # FUNCTION [protected]: save model weights
    def __save_model(self, model, epochs: int, loss: float, check_type: str = ""):

        torch.save({'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss},
                   self.model_save + check_type)

    # FUNCTION: train step
    def _train_step(self, model, tr_loader, ref_loader, fixed_anchor: bool):

        # set model to training mode
        model.train()

        # training metrics
        total_loss, num_mined_triplets, batch_counter = 0.0, 0, 0

        # unpack reference samples
        ref_batch = next(iter(ref_loader))
        ref_batch = move_batch_to_device(ref_batch, device=self.device)
        ref_label = ref_batch.y

        # iterate over all batches
        for batch_id, batch in enumerate(tr_loader):

            # zeroing out all gradients
            self.optimizer.zero_grad()

            # cast data onto devices
            batch = move_batch_to_device(batch, device=self.device)
            y_true = batch.y

            # transform reference samples into embeddings
            ref_emb = model(ref_batch)

            # get batch embeddings from the model
            embeddings = model(batch)

            # determine if miner was assigned
            if self.miner is not None:
                mined_indices_tuple = self.miner(embeddings, y_true)

                if fixed_anchor:
                    loss = self.loss_func(embeddings[:-1], y_true[:-1], mined_indices_tuple)
                else:
                    loss = self.loss_func(embeddings, y_true, mined_indices_tuple)

            else:
                loss = self.loss_func(embeddings, y_true, ref_emb=ref_emb, ref_labels=ref_label)

            # backward prop and optim
            loss.backward()
            self.optimizer.step()

            if self.loss_optimizer is not None:
                self.loss_optimizer.step()

            # update trained metrics
            total_loss += loss.item()

            try:
                num_mined_triplets += self.miner.num_triplets

            except Exception as e:
                pass

            # increase batch counter
            batch_counter += 1

        # track epoch loss
        epoch_loss = round(total_loss / batch_counter, 4)

        # handle reference embeddings after final model weight update
        with torch.no_grad():
            model.eval()
            ref_emb = model(ref_batch)

        # detach from tensor
        ref_out = ref_emb.detach().cpu().numpy()
        ref_label = ref_label.detach().cpu().numpy()

        return epoch_loss, num_mined_triplets, ref_out, ref_label

    # main training function looping through all epochs
    def train_loop(self, model, tr_loader, ref_loader, num_epochs: int, k: int, fixed_anchor: bool = False):

        # output container
        embeddings, labels, loss_stats, early_stop_ctr = [], [], [], 0

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
                ep_tr_loss, num_triplets, ref_embs, ref_labels = self._train_step(model=model,
                                                                                  tr_loader=tr_loader,
                                                                                  ref_loader=ref_loader,
                                                                                  fixed_anchor=fixed_anchor)

                # appending
                loss_stats.append(ep_tr_loss)

                print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}] ==> tr_loss: {ep_tr_loss:.5f},"
                      f" num of mined training triplets per epoch: {num_triplets}")

                self.scheduler.step(ep_tr_loss)

                if ep_tr_loss < self.loss_threshold:

                    print(f"[FOLD 0{k}]: Epoch [{epoch+1}|{num_epochs}]: loss threshold undercut {ep_tr_loss} < {self.loss_threshold}\n")
                    self.__save_model(model=model, epochs=epoch+1, loss=ep_tr_loss, check_type=f"_check.pth")

                    return ref_embs, ref_labels, loss_stats, epoch

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

                        self.__save_model(model=model, epochs=num_epochs, loss=ep_tr_loss, check_type=f"_check.pth")
                        print(f"[FIN]: Training complete ==> saved to [{self.model_save}]...")

                        return ref_embs, ref_labels, loss_stats, epoch

            # after final epoch
            self.__save_model(model=model, epochs=num_epochs, loss=ep_tr_loss, check_type=f"_final.pth")
            print(f"[FIN]: Training complete ==> saved to [{self.model_save}]...")

            return ref_embs, ref_labels, loss_stats, epoch

        except KeyboardInterrupt:
            print("[KEYBOARD INTERRUPT]: exiting from training because of STRG+C!")


