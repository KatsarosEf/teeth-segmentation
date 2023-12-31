import os
import torch
import wandb
from argparse import ArgumentParser
from utils.dataset import MTL_Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from losses import SemanticSegmentationLoss
from metrics import SegmentationMetrics
from utils.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
from utils.network_utils import model_save, model_load
from models import UnetPlusPlus

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def train(args, dataloader, model, optimizer, scheduler, losses_dict, metrics_dict, epoch):
    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    losses_cumltive = {k: [] for k in tasks}
    model.train()
    for seq_num, seq in enumerate(dataloader):
        for frame in range(args.prev_frames, args.seq_len):
            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
            else:
                frames.append(seq['image'][frame].cuda(non_blocking=True))
                del frames[0]
            #
            # gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
            #                 [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}
            #
            # # Compute model predictions, errors and gradients and perform the update
            # optimizer.zero_grad()
            # outputs = model(frames[0])
            # outputs = {"segment_one": outputs,
            #            "segment_two": outputs,
            #            "segment_three": outputs}
            #
            #
            # losses = {task: losses_dict[task](outputs[task], gt_dict[task]) for task in tasks}
    #         loss = sum(losses.values())
    #         loss.backward()
    #         #torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # added gradient clipping and normalization
    #         optimizer.step()
    #
    #         print('[TRAIN] [EPOCH:{}/{} ] [SEQ: {}/{}] Total Loss: {:.4f}\t{}'.format(epoch, args.epochs, seq_num+1, len(dataloader), loss, '\t'.join(
    #             ['{} loss: {:.4f}'.format(task, losses[task]) for task in tasks])))
    #
    #         # Compute metrics for the tasks at hand
    #         task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
    #         metrics_values = {k: v.item() for task in tasks for k, v in task_metrics[task].items()}
    #
    #         print("[TRAIN] [EPOCH:{}/{} ] {}".format(epoch, args.epochs, '\t'.join(
    #             ['{}: {:.4f}'.format(k, metrics_values[k]) for k in metrics])))
    #         for metric in metrics:
    #             metric_cumltive[metric].append(metrics_values[metric])
    #         for t in tasks:
    #             losses_cumltive[t].append(float(losses[t]))
    #
    # scheduler.step()
    # wandb_logs = {"Train - {}".format(m): sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    # for t in tasks:
    #     wandb_logs['Loss: {}'.format(t)] = sum(losses_cumltive[t])/len(losses_cumltive[t])
    # wandb_logs['epoch'] = epoch
    # wandb_logs['lr'] = optimizer.param_groups[0]['lr']
    # wandb.log(wandb_logs)



def val(args, dataloader, model, metrics_dict, epoch):
    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    model.eval()

    with torch.no_grad():

        for seq_idx, seq in enumerate(dataloader):
            for frame in range(args.prev_frames, args.seq_len):
                # Load the data and mount them on cuda
                if frame == args.prev_frames:
                    frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
                else:
                    frames.append(seq['image'][frame].cuda(non_blocking=True))
                    del frames[0]

                gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
                [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}

                outputs = model(frames[0])
                outputs = dict(zip(tasks, outputs))

                task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
                metrics_values = {k: v.item() for task in tasks for k, v in task_metrics[task].items()}

                for metric in metrics:
                    metric_cumltive[metric].append(metrics_values[metric])


        metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}

        print("\n[VALIDATION] [EPOCH:{}/{}] {}\n".format(epoch, args.epochs,
                                                         ' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))
        wandb_logs = {"Val - {}".format(m): metric_averages[m] for m in metrics}
        wandb_logs['epoch'] = epoch
        wandb.log(wandb_logs)


def main(args):


    tasks = [task for task in ['segment_one', 'segment_two', 'segment_three'] if getattr(args, task)]

    transformations = {'train': transforms.Compose([
        # ColorJitter(), RandomHorizontalFlip(),RandomVerticalFlip(),
        ToTensor(), Normalize()]),
                       'val': transforms.Compose([ToTensor(), Normalize()])}

    data = {split: MTL_Dataset(tasks, args.data_path, split, args.seq_len, transform=transformations[split])
            for split in ['train', 'val']}

    loader = {split: DataLoader(data[split], batch_size=args.bs, shuffle=split == 'train', num_workers=4, pin_memory=True)
              for split in ['train', 'val']}

    losses_dict = {
        'segment_one': SemanticSegmentationLoss().cuda(),
        'segment_two': SemanticSegmentationLoss().cuda(),
        'segment_three': SemanticSegmentationLoss().cuda()

    }
    losses_dict = {k: v for k, v in losses_dict.items() if k in tasks}

    metrics_dict = {
        'segment_one': SegmentationMetrics().cuda(),
        'segment_two': SegmentationMetrics().cuda(),
        'segment_three': SegmentationMetrics().cuda()

    }
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks if k in tasks}


    model = UnetPlusPlus(tasks, classes=3).cuda()

    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.resume:
        checkpoint_file_name = 'ckpt_{}.pth'.format(args.resume_epoch) if args.resume_epoch else 'ckpt.pth'
        resume_path = os.path.join(args.out, 'models', checkpoint_file_name)
        resume_epoch = model_load(resume_path, model, optimizer, scheduler)
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 1
        if 'debug' in args.out:
            os.makedirs(os.path.join(args.out, 'models'), exist_ok=True)
        else:
            os.makedirs(os.path.join(args.out, 'models'))

    wandb.init(project='mtl-normal', name='dst-cv', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)



    for epoch in range(start_epoch, args.epochs+1):

        train(args, loader['train'], model, optimizer, scheduler, losses_dict, metrics_dict, epoch)

        val(args, loader['val'], model, metrics_dict, epoch)

        model_save(model, optimizer, scheduler, epoch, args)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default="/media/efklidis/4TB/medical-annotations-release", type=str)
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/debug', type=str)

    # parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/dblab_real', type=str) #/media/efklidis/4TB/ # ../raid/data_ours_new_split
    # parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/debug', type=str)

    parser.add_argument('--resume_epoch', dest='resume_epoch', help='Number of epoch to resume', default=0, type=int)
    parser.add_argument("--segment_one", action='store_false', help="Flag for segmentation")
    parser.add_argument("--segment_two", action='store_false', help="Flag for segmentation")
    parser.add_argument("--segment_three", action='store_false', help="Flag for segmentation")

    parser.add_argument("--resume", action='store_true', help="Flag for resume training")
    parser.add_argument('--model', dest='model', help='Set type of model', default='deeplab', type=str)

    parser.add_argument('--epochs', dest='epochs', help='Set number of epochs', default=50, type=int)
    parser.add_argument('--bs', help='Set size of the batch size', default=4, type=int)
    parser.add_argument('--lr', help='Set learning rate', default=1e-4, type=float)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=1, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=0, type=int)

    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)


    args = parser.parse_args()

    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(args)
