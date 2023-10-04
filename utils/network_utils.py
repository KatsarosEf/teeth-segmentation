import torch
import os
import time
import fvcore.nn.flop_count as flop_count
from fvcore.nn import FlopCountAnalysis



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def model_save(model, optimizer, scheduler, epoch, args, save_best=False):

    save_dict = {
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    torch.save(save_dict, os.path.join(args.out, 'models/ckpt.pth'))

    if save_best:
        torch.save(save_dict, os.path.join(args.out, 'models/best_ckpt.pth'))

    if epoch % args.save_every == 0:
        torch.save(save_dict, os.path.join(args.out, 'models/ckpt_{}.pth'.format(epoch)))


def model_load(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']

    return epoch


def measure_efficiency(args):
    from models.deepLabv3Plus import DeepLabv3Plus
    from models.unetPlusPlus import UnetPlusPlus

    tasks = ["segment"]
    # model = Model(tasks).cuda()
    model = DeepLabv3Plus(tasks).cuda()
    model.eval()
    # model = DeepLabv3Plus(tasks).cuda()
    # model.eval()
    dims = 800, 800
    x1 = torch.randn((1, 3, *dims)).cuda(non_blocking=True)
    print(x1.shape)
    times = []
    with torch.no_grad():
        for i in range(100):
            torch.cuda.synchronize()
            test_time_start = time.time()
            _ = model.forward(x1)
            torch.cuda.synchronize()
            times.append(time.time() - test_time_start)

    fps = round(1 / (sum(times[30:])/len(times[30:])), 2)
    params = count_parameters(model) / 10 ** 6
    flops = FlopCountAnalysis(model, (x1)).total() * 1e-9
    del model
    return params, fps, flops

