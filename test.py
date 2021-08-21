from train import get_args_parser
from timm.models import create_model
from tools.calculate_tool import matrixs, AucCal
from tools.data_loader import *
from tqdm.auto import tqdm


@torch.no_grad()
def test(mm, test_loader):
    all_pre = None
    all_true = None
    hehe = 0
    for i_batch, sample_batch in enumerate(tqdm(test_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        outputs = mm(inputs)
        if args.multi_label:
            outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        if hehe == 0:
            all_pre = outputs
            all_true = labels
            hehe = 1
        else:
            all_pre = np.concatenate((all_pre, outputs), axis=0)
            all_true = np.concatenate((all_true, labels), axis=0)
    if not args.multi_label:
        preds = np.argmax(all_pre, axis=1)
        matrixs(preds, all_true, model_name)
    else:
        epoch_auc = AucCal(make_graph=True).cal_auc(all_pre, all_true, model_name)
        print(":", epoch_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = "efficientnet_b2_checkpoint.pth" #需要和train的model对应
    device = torch.device(args.device)
    if args.multi_label:
        args.num_class = 4
    model = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_class)
    model.to(device)
    train, val = MakeList(args).read_csv()
    dataset_val = CityDataset(val, args, transform=make_transform("val"))
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, num_workers=args.num_workers)
    checkpoint = torch.load("saved_model/" + model_name)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test(model, data_loader_val)