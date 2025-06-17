import numpy as np
import torch
    
class TransBuffer:
    def __init__(self, obs_dim, buffer_size=20_000):
        self.obs_buf = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.out_buf = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)

        self.idx = 0  # Current write index
        self.size = 0  # Current size of the buffer
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim

    def store(self, obs_batch, out_batch):
        # Ensure the input arrays have the correct shape
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(-1, self.obs_dim)
        if out_batch.ndim == 1:
            out_batch = out_batch.reshape(-1, self.obs_dim)

        batch_size = obs_batch.shape[0]
        if batch_size + self.idx > self.buffer_size:
            # Split the batch if it exceeds the buffer size
            split_idx = self.buffer_size - self.idx
            self.obs_buf[self.idx:self.buffer_size] = obs_batch[:split_idx]
            self.out_buf[self.idx:self.buffer_size] = out_batch[:split_idx]
            self.obs_buf[0:batch_size - split_idx] = obs_batch[split_idx:]
            self.out_buf[0:batch_size - split_idx] = out_batch[split_idx:]
        else:
            self.obs_buf[self.idx:self.idx + batch_size] = obs_batch
            self.out_buf[self.idx:self.idx + batch_size] = out_batch

        # Update index and size
        self.idx = (self.idx + batch_size) % self.buffer_size
        self.size = np.minimum(self.size + batch_size, self.buffer_size)

    def sample_batch(self, batch_size=1000):
        idxs = np.random.choice(self.size, min(self.size, batch_size), replace=False)
        obs_batch = self.obs_buf[idxs]
        out_batch = self.out_buf[idxs]
        return dict(obs=obs_batch, out=out_batch)

    def sample_all(self):
        return dict(obs=self.obs_buf[:self.size], out=self.out_buf[:self.size])