import torch
from tqdm import tqdm
from torch.optim import Adam
from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule

from pathfold.model.model_frame import Denoiser


class Diffusion(LightningModule, ABC):

	def __init__(self, config):
		super(Diffusion, self).__init__()

		self.config = config

		self.model = Denoiser(
			**self.config.model,
			n_timestep=self.config.diffusion['n_timestep']
		)

		self.setup = False

	@abstractmethod
	def setup_schedule(self):
		'''
		Set up variance schedule and precompute its corresponding terms.
		'''
		raise NotImplemented

	@abstractmethod
	def transform(self, batch):
		'''
		Transform batch data from data pipeline into the desired format

		Input:
			batch - coordinates from data pipeline (shape: b x (n_res * 3))

		Output: frames (shape: b x n_res)
		'''
		raise NotImplemented

	@abstractmethod
	def sample_timesteps(self, num_samples):
		raise NotImplemented

	@abstractmethod
	def sample_frames(self, mask):
		raise NotImplemented

	@abstractmethod
	def q(self, t0, s, mask):
		raise NotImplemented

	@abstractmethod
	def p(self, ts, s, mask, noise_scale):
		raise NotImplemented

	@abstractmethod
	def loss_fn(self, tnoise, ts, s, single=None, pair=None):
		raise NotImplemented

	def p_sample_loop(self, mask, noise_scale, verbose=True, single=None, pair=None, frame=None):
		if not self.setup:
			self.setup_schedule()
			self.setup = True
		ts = self.sample_frames(mask)
		ts_seq = [ts]
		#print('steps', self.config.diffusion['n_timestep'])
		for i in tqdm(reversed(range(self.config.diffusion['n_timestep'])), desc='sampling loop time step', total=self.config.diffusion['n_timestep'], disable=not verbose):
			s = torch.Tensor([i] * mask.shape[0]).long().to(self.device)
			ts = self.p(ts, s, mask, noise_scale, single, pair, frame)
			ts_seq.append(ts)
		return ts_seq

	def training_step(self, batch, batch_idx):
		'''
		Training iteration.

		Input:
			batch     - coordinates from data pipeline (shape: b x (n_res * 3))
			batch_idx - batch index (shape: b)

		Output: Either a single loss value or a dictionary of losses, containing
			one key as 'loss' (loss value for optimization)
		'''
		if not self.setup:
			self.setup_schedule()
			self.setup = True
		#update
		t0, mask, single, pair, frame = self.transform(batch)
		s = self.sample_timesteps(t0.shape[0])
		ts, tnoise = self.q(t0, s, mask)
		#update
		loss = self.loss_fn(tnoise, ts, s, mask, single, pair, frame)
		self.log('train_loss', loss, on_step=True, on_epoch=True)
		return loss

	def configure_optimizers(self):
		return Adam(
			self.model.parameters(),
			lr=self.config.optimization['lr']
		)