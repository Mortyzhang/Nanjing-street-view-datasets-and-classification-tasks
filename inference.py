from train import get_args_parser
from timm.models import create_model
import torch
from tools.data_loader import *
import csv
from tqdm.auto import tqdm


@torch.no_grad()
def test(mm, test_loader):
    record = []
    for i_batch, sample_batch in enumerate(tqdm(test_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        name_list = sample_batch["names"][0].split("/")
        name = name_list[-1][:-4]
        outputs = mm(inputs)
        if args.multi_label:
            outputs = torch.sigmoid(outputs)
            outputs = outputs.cpu().numpy()
            outputs = np.concatenate([1-outputs, outputs], axis=0)
            final = np.argmax(outputs, axis=0)
        else:
            _, pred = torch.max(outputs, dim=1)
            final = pred[0].cpu().numpy()

        record.append([name, final])
    return record


def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8",newline='')
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["ID", "Category"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    args.inference = True
    args.batch_size = 1
    args.num_workers = 0
    if args.multi_label:
        args.num_class = 4
    model = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_class)
    model.to(device)
    all_img = MakeListInference(args).load_folder_file()
    print("-----------------------")
    dataset_val = CityDataset(all_img, args, transform=make_transform("inference"))
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, num_workers=args.num_workers)
    checkpoint = torch.load("saved_model/inception_4__checkpoint.pth") # load model需要和trian的model对应
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out = test(model, data_loader_val)
    make_csv(out, f"{'_multi_' if args.multi_label else ''}" + args.model)