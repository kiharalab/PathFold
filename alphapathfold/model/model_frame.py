import torch
from torch import nn

from alphapathfold.model.single_feature_net import SingleFeatureNet
from alphapathfold.model.pair_feature_net import PairFeatureNet
from alphapathfold.model.pair_transform_net import PairTransformNet
from alphapathfold.model.structure_net import StructureNet

from alphapathfold.utils.encoding import sinusoidal_encoding

from alphapathfold.model.conlstm import ConvLSTM
#update
class LabelEmbedder(nn.Module):
	"""
	Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
	"""
	def __init__(self, num_classes, hidden_size, dropout_prob):
		super().__init__()
		self.embedding_table = nn.Embedding(num_classes, hidden_size)
		self.num_classes = num_classes
		self.dropout_prob = dropout_prob

	def forward(self, labels):
		# print('token drop labels:', labels, labels.shape)
		embeddings = self.embedding_table(labels)
		return embeddings
	
class ContactBlock2D(nn.Module):
	def __init__(self, channels=128, kernel_size=3, padding=1, dropout=0.1, stride=1, dilation=1, in_channels=1):

		super(ContactBlock2D, self).__init__()

		padding = padding * dilation
		
		self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, stride, padding, dilation)
		self.bn1 = nn.InstanceNorm2d(channels, affine=True)
		self.elu1 = nn.ELU()
		self.dropout1 = nn.Dropout2d(p=dropout)

		self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
		self.bn2 = nn.InstanceNorm2d(channels, affine=True)
		self.elu2 = nn.ELU()
		
	def forward(self, x):
		#print('x', x.size())
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.elu1(out)

		out = self.dropout1(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out = self.elu2(out)
		#print('out', out.size())
		return out
	
class Denoiser(nn.Module):

	def __init__(self,
		c_s, c_p, n_timestep,
		c_pos_emb, c_timestep_emb,
		relpos_k, template_type,
		n_pair_transform_layer, include_mul_update, include_tri_att,
		c_hidden_mul, c_hidden_tri_att, n_head_tri, tri_dropout, pair_transition_n,
		n_structure_layer, n_structure_block,
		c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout,
		n_structure_transition_layer, structure_transition_dropout,
		contact_in_channels=3
	):
		super(Denoiser, self).__init__()

		self.single_feature_net = SingleFeatureNet(
			c_s,
			n_timestep,
			c_pos_emb,
			c_timestep_emb
		)
		
		self.pair_feature_net = PairFeatureNet(
			c_s,
			c_p,
			relpos_k,
			template_type
		)

		self.pair_transform_net = PairTransformNet(
			c_p,
			n_pair_transform_layer,
			include_mul_update,
			include_tri_att,
			c_hidden_mul,
			c_hidden_tri_att,
			n_head_tri,
			tri_dropout,
			pair_transition_n
		) if n_pair_transform_layer > 0 else None

		self.structure_net = StructureNet(
			c_s,
			c_p,
			n_structure_layer,
			n_structure_block,
			c_hidden_ipa,
			n_head_ipa,
			n_qk_point,
			n_v_point,
			ipa_dropout,
			n_structure_transition_layer,
			structure_transition_dropout
		)

		self.single_project = nn.Linear(384, 128)
		self.pair_project = nn.Linear(128, 128)
		# self.frame_project_i = nn.Sequential(
		# 						nn.Linear(3,128),
		# 						nn.ReLU(),
		# 						nn.Linear(128,128),
		# 						nn.ReLU(),
		# 					)

		# self.frame_project_j = nn.Sequential(
		# 						nn.Linear(3,128),
		# 						nn.ReLU(),
		# 						nn.Linear(128,128),
		# 						nn.ReLU(),
		# 					)
		self.contact_res = ContactBlock2D(in_channels=contact_in_channels)
			
		# self.covlstm = ConvLSTM(input_dim=1,
		# 					hidden_dim=[128, 128],
		# 					kernel_size=(3, 3),
		# 					num_layers=2,
		# 					batch_first=True,
		# 					bias=True, return_all_layers=False)
		  
		#self.label_embedder = LabelEmbedder(10, 128, 0.1)

	def forward(self, ts, timesteps, mask, single=None, pair=None, frame=None):
		p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

		s = self.single_feature_net(ts, timesteps, mask)
		p = self.pair_feature_net(s, ts, p_mask)

		b, max_n_res, device = ts.shape[0], ts.shape[1], timesteps.device
		# # # [b, n_res, c_pos_emb]
		# frame_encode_i = self.frame_project_i(frame.float())
		# frame_encode_j = self.frame_project_j(frame.float())

		# #print(frame_encode.size())
		# frame_encode_i = torch.tile(frame_encode_i.unsqueeze(1), (1, s.shape[1], 1))
		# frame_encode_j = torch.tile(frame_encode_j.unsqueeze(1), (1, s.shape[1], 1))

		# frame_pair = frame_encode_i[:, :, None, :] + frame_encode_j[:, None, :, :]
		# frame_pair *= p_mask.unsqueeze(-1)
		# #update
		# #s = s + frame_encode
		# p = p + frame_pair
		
		# frame_encode = self.label_embedder(frame)
		# #print(frame_encode.size())
		
		# frame_encode = frame_encode.repeat(1, max_n_res, 1)
		# #print(frame_encode.size())
		# frame_encode = frame_encode * mask.unsqueeze(-1)
		# s = s + frame_encode
		#print(frame.size())
		s = s + self.single_project(single.float())
		p = p + self.pair_project(pair.float())
		
		#time-based feature fusion
		# _, last_states = self.covlstm(frame.unsqueeze(2))
		# time_based = last_states[0][0]
		# p = p + self.contact_res(frame).permute(0, 2, 3, 1) + time_based.permute(0, 2, 3, 1) 
			
		#regular convolution feature
		p = p + self.contact_res(frame).permute(0, 2, 3, 1)

		if self.pair_transform_net is not None:
			p = self.pair_transform_net(p, p_mask)
		ts = self.structure_net(s, p, ts, mask)
		return ts
