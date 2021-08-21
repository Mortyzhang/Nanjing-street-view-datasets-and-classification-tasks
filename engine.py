import torch
from tools.calculate_tool import evaluateTop1, AucCal
from tqdm.auto import tqdm
import numpy as np


def train_one_epoch(args, model, optimizer, data_loader, device, criterion, record, epoch):
    model.train()
    L = len(data_loader)
    running_loss = 0.0
    if args.multi_label:
        all_pre = None
        all_true = None
        hehe = 0
    else:
        running_corrects = 0.0

    print("start training epoch: " + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        # zero the gradient parameter
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        a = loss.item()
        running_loss += a
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}  LR: {}".format(epoch, i_batch, L-1, a, optimizer.param_groups[0]["lr"]))

        if args.multi_label:
            preds = torch.sigmoid(logits)
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if hehe == 0:
                all_pre = preds
                all_true = labels
                hehe = 1
            else:
                all_pre = np.concatenate((all_pre, preds), axis=0)
                all_true = np.concatenate((all_true, labels), axis=0)
        else:
            running_corrects += evaluateTop1(logits, labels)

    epoch_loss = round(running_loss/L, 3)
    record["train"]["loss"].append(epoch_loss)
    if args.multi_label:
        epoch_auc = AucCal().cal_auc(all_pre, all_true)
        record["train"]["auc"].append(epoch_auc)
    else:
        epoch_acc_1 = round(running_corrects/L, 3)
        record["train"]["acc"].append(epoch_acc_1)


@torch.no_grad()
def evaluate(args, model, data_loader, device, criterion, record, epoch):
    model.eval()
    L = len(data_loader)
    running_loss = 0.0
    if args.multi_label:
        all_pre = None
        all_true = None
        hehe = 0
    else:
        running_corrects = 0.0

    print("start evaluate  epoch: " + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

        logits = model(inputs)
        loss = criterion(logits, labels)

        a = loss.item()
        running_loss += a
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))

        if args.multi_label:
            preds = torch.sigmoid(logits)
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if hehe == 0:
                all_pre = preds
                all_true = labels
                hehe = 1
            else:
                all_pre = np.concatenate((all_pre, preds), axis=0)
                all_true = np.concatenate((all_true, labels), axis=0)
        else:
            running_corrects += evaluateTop1(logits, labels)

    epoch_loss = round(running_loss/L, 3)
    record["val"]["loss"].append(epoch_loss)
    if args.multi_label:
        epoch_auc = AucCal().cal_auc(all_pre, all_true)
        record["val"]["auc"].append(epoch_auc)
    else:
        epoch_acc_1 = round(running_corrects/L, 3)
        record["val"]["acc"].append(epoch_acc_1)