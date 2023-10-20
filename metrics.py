import torch
from torch.nn import Module


def binary_segmentation_postprocessing(output: torch.Tensor):

	N, C, H, W = output.shape
	processed = torch.zeros((N, 1, H, W), dtype=torch.long).to(output.device)

	processed[output[:, 1:2] - output[:, 0:1] > 0] = 1
	return processed


def iou_score_binary(outputs: torch.Tensor, labels: torch.Tensor):
	e = 1e-6
	intersection = (outputs & labels).float().sum((-1, -2)) + e
	union = (outputs | labels).float().sum((-1, -2)) + e

	iou = intersection / union
	return iou.mean()


def dice_score_binary(outputs: torch.Tensor, labels: torch.Tensor):
	e = 1e-6
	intersection = (outputs & labels).float().sum((-1, -2)) + e
	union = (outputs | labels).float().sum((-1, -2)) + e

	dice = 2 * intersection / (intersection + union)
	return dice.mean()


class Metric(Module):

	def __init__(self):
		super(Metric, self).__init__()

	def forward(self, output, gt):
		with torch.no_grad():
			metric_value = self.compute_metric(output, gt)
		return metric_value

	def compute_metric(self, output, gt):
		return None


class IoU(Metric):

	def compute_metric(self, output, gt):
		return iou_score_binary(output, gt)


class Dice(Metric):

	def compute_metric(self, output, gt):
		return dice_score_binary(output, gt)



class SegmentationMetrics(Module):

	def __init__(self):
		super(SegmentationMetrics, self).__init__()
		self.metrics = {
			'iou': IoU(),
			'dice': Dice()
		}

	def forward(self, output, gt):
		with torch.no_grad():
			output = torch.argmax(output, 1)
			metric_results = {metric: metric_function(output, gt) for metric, metric_function in self.metrics.items()}
		return metric_results


if __name__ == '__main__':
	prediction = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[1,1,0],[0,0,0]]])
	gt = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]], [[0,1,0],[0,1,0],[0,0,0]]])

	segmentation_metrics = SegmentationMetrics()
	print(segmentation_metrics(prediction, gt))

