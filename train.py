import argparse
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
import tools.prepare_things as prt
from engine import train_one_epoch, evaluate
from tools.data_loader import *
from timm.models import create_model
import datetime
import time


def get_args_parser():
    parser = argparse.ArgumentParser('Set classification model', add_help=False)
    parser.add_argument('--model', default="efficientnet_b2", type=str,
                        help="check model type in https://github.com/rwightman/pytorch-image-models")
    # using model name in timm\models resnest50d这里的model需要和inference和test的model对应
    # training set
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument("--num_class", default=5, type=int)
    parser.add_argument('--img_size', default=260, help='size of input image')
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument('--pre_trained', default=True, help='whether use pre parameter for backbone')
    parser.add_argument("--multi_label", default=False, type=bool) #show multilabel

    # inference set
    parser.add_argument('--inference', default=False, type=str)
    parser.add_argument('--inference_dir', default="city_analysis/", type=str) #processing image location处理的图片路径

    # data/machine set
    parser.add_argument('--dataset_dir', default='city_analysis/',
                        help='path for save data')
    parser.add_argument('--csv_root', default='data_profile.csv', #train_CSV路径
                        help='path csv record')
    parser.add_argument('--output_dir', default='saved_model/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    record = {"train": {"loss": []},
              "val": {"loss": []}}
    if not args.multi_label:
        criterion = torch.nn.CrossEntropyLoss()
        record["train"].update({"acc": []})
        record["val"].update({"acc": []})
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss()
        args.num_class = 4
        record["train"].update({"auc": []})
        record["val"].update({"auc": []})

    model = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_class)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9, lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError(f'unknown {args.optimizer}')

    print(args.num_class)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    train, val = MakeList(args).read_csv()
    dataset_train = CityDataset(train, args, transform=make_transform("train"))
    dataset_val = CityDataset(val, args, transform=make_transform("val"))
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoaderX(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    best = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(args, model, optimizer, data_loader_train, device, criterion, record, epoch)
        lr_scheduler.step()
        evaluate(args, model, data_loader_val, device, criterion, record, epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / (args.model + f"{'_multi_' if args.multi_label else ''}" + '_checkpoint.pth')]
            # extra checkpoint before LR drop and every 100 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
            if args.multi_label:
                if record["val"]["auc"][epoch] > best:
                    best = record["val"]["auc"][epoch]
                    checkpoint_paths.append(output_dir / (args.model + f"{'_multi' if args.multi_label else ''}" + f'_checkpoint{epoch:04}.pth'))
            else:
                if record["val"]["acc"][epoch] > best:
                    best = record["val"]["acc"][epoch]
                    checkpoint_paths.append(output_dir / (args.model + f"{'_multi' if args.multi_label else ''}" + f'_checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                prt.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        print("train loss:", record["train"]["loss"])
        print("val loss:", record["val"]["loss"])
        if not args.multi_label:
            print("train acc:", record["train"]["acc"])
            print("val acc:", record["val"]["acc"])
        else:
            print("train auc:", record["train"]["auc"])
            print("val auc:", record["val"]["auc"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)