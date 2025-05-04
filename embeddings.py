import numpy as np
import torch


class Embedding:

    def __init__(self, model, device):
        super(Embedding, self).__init__()

        self.model = model
        self.device = device

    @torch.no_grad()
    def embed_batch(self, batch_data):

        self.model.eval()

        # output container
        embeddings = []

        with torch.no_grad():
            for batch_nr, batch in enumerate(batch_data):

                # compute embeddings
                _, _, batch_emb = self.model(batch.to(self.device))
                embeddings.extend(batch_emb.cpu().numpy())

            return np.asarray(embeddings)

    @torch.no_grad()
    def embed_sample(self, data):

        self.model.eval()

        with torch.no_grad():
            sample = next(iter(data))[0]
            _, _, embedding = self.model(sample.to(self.device))

        return embedding.cpu().numpy()
