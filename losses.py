import torch
from torch import nn
import torch.nn.functional as F

class SemanticSegmentationLoss(nn.Module):

	def __init__(self, ce_factor=1, device='cuda'):
		super(SemanticSegmentationLoss, self).__init__()
		self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
		self.ce_factor = ce_factor

	def forward(self, output, gt):
		cross_entropy_loss = self.cross_entropy_loss(output[0], gt)
		return self.ce_factor * cross_entropy_loss

class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, self).__init__()
		k = torch.Tensor([[.05, .25, .4, .25, .05]])
		self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
		if torch.cuda.is_available():
			self.kernel = self.kernel.cuda()
		self.loss = CharbonnierLoss()

	def conv_gauss(self, img):
		n_channels, _, kw, kh = self.kernel.shape
		img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
		return F.conv2d(img, self.kernel, groups=n_channels)

	def laplacian_kernel(self, current):
		filtered    = self.conv_gauss(current)    # filter
		down        = filtered[:,:,::2,::2]               # downsample
		new_filter  = torch.zeros_like(filtered)
		new_filter[:,:,::2,::2] = down*4                  # upsample
		filtered    = self.conv_gauss(new_filter) # filter
		diff = current - filtered
		return diff

	def forward(self, x, y):
		loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
		return loss

class MSEdgeLoss(nn.Module):

	def __init__(self, device='cuda'):
		super(MSEdgeLoss, self).__init__()
		self.loss_function = EdgeLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.loss_function(elem, torch.nn.functional.interpolate(gt, scale_factor=1.0/(2**num))))
			loss = sum(losses)
		else:
			loss = self.loss_function(output, gt)
		return loss


class CharbonnierLoss(nn.Module):

	def __init__(self):
		super(CharbonnierLoss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		return torch.sqrt((X - Y) ** 2 + self.eps).sum()



