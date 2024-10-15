import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utility import VPROM_dataset_utility
from utils import initialize_wandb, VPROM_splits

from savirt import SAViRT

parser = argparse.ArgumentParser(description="savir-t")
parser.add_argument("--name", type=str, default="savir-t_VPROM_2")
parser.add_argument("--model", type=str, default="SAViR-T", choices=["SAViR-T"])
parser.add_argument(
    "--dataset", type=str, default="V-PROM_FEATURES", choices=["V-PROM_FEATURES"]
)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--feature_size", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--load_workers", type=int, default=16)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--train_data", type=str, default="/data/local/datasets/features/all/train/", help='Replace with your local folder; containing the V-RPOM training images extracted ResNet101 features.')
parser.add_argument("--test_data", type=str, default="/data/local/datasets/features/all/test/", help='Replace with your local folder; containing the V-RPOM testing images extracted ResNet101 features.')
parser.add_argument("--save", type=str, default="checkpoints")
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument("--meta_beta", type=float, default=0.0)
parser.add_argument("--use_cell", type=float, default=1.0)
parser.add_argument("--use_ind", type=float, default=1.0)
parser.add_argument("--use_eco", type=float, default=1.0)
parser.add_argument("--cuda", default=True)
parser.add_argument("--use_wandb", default=False)
parser.add_argument('--tag', type=int, default=1)


args = parser.parse_args()
torch.cuda.manual_seed(args.seed)

random.seed(args.seed)
args.save += "/" + args.dataset + "/"
start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
args.save += args.name + "/"
if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.use_wandb:
    wandb = initialize_wandb(args)

fname_imgs_test, fname_imgs_train, fname_targets_test, fname_targets_train = VPROM_splits()

train = VPROM_dataset_utility(
    flag = 1,
    fnames_imgs=fname_imgs_train,
    fnames_target=fname_targets_train,
    img_size=[3,args.img_size,args.img_size],
    M=16,
    shuffle=True,
    train_folder=args.train_data
)

test = VPROM_dataset_utility(
    flag = 0,
    fnames_imgs=fname_imgs_test,
    fnames_target=fname_targets_test,
    img_size=[3,args.img_size,args.img_size],
    M=16,
    test_folder=args.test_data
)

trainloader = DataLoader(
    train, batch_size=args.batch_size, shuffle=True, num_workers=args.load_workers, drop_last = True
)
testloader = DataLoader(
    test, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers, drop_last = False
)

print("Dataset:", args.dataset)
print("Train/Test:{0}/{1}".format(len(train), len(test)))
print("Image size:", args.img_size)

model = SAViRT(args)

start_epoch = 0
minibatch_idx = 0
if args.resume:
    args.resume_epoch = 20
    model.load_model(args.resume, args.resume_epoch)
    print("Loaded model")
    start_epoch = args.resume_epoch + 1

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

writer = SummaryWriter(os.path.join(args.log_dir, args.name, "runs"))

with open(os.path.join(args.save, "results.log"), "w") as f:
    for key, value in vars(args).items():
        f.write("{0}: {1}\n".format(key, value))
    f.write("--------------------------------------------------\n")

model = model.cuda()

def train(epoch):
    global minibatch_idx
    model.train()
    train_loss = 0
    accuracy = 0
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    torch.cuda.empty_cache()
    for batch_idx, (image, target, frames) in enumerate(trainloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
        model.optimizer.zero_grad()
        output = model(image)
        loss = model.compute_loss(output, target)
        loss.backward()
        model.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        loss, acc = loss.item(), accuracy
        if (batch_idx + 1) % 100 == 0:
            print(
                "Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.".format(
                    epoch, batch_idx, loss, acc
                )
            )            
        loss_all += loss
        acc_all += acc
        minibatch_idx += 1
        torch.cuda.empty_cache()
    if counter > 0:
        writer.add_scalar("Training_acc", acc_all / float(counter), minibatch_idx)
    return loss_all / float(counter), acc_all / float(counter)

def test(epoch):
    model.eval()
    acc = []
    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, frames) in enumerate(testloader):
            counter += 1
            if args.cuda:
                image = image.cuda()
                target = target.cuda()
            output = model(image)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).cpu().numpy() * 100
            acc.append(correct)
    if counter > 0:
        acc_avg = np.concatenate(acc).mean()
        print("Total Testing Acc: {:.4f}".format(acc_avg))
        writer.add_scalar("Testing_acc", acc_avg, minibatch_idx)
    return acc_avg

for epoch in range(start_epoch, args.epochs):
    avg_train_loss, avg_train_acc = train(epoch)
    avg_test_acc = test(epoch)
    if epoch % 20 == 0:
        model.save_model(args.save, epoch)
    with open(os.path.join(args.save, "results.log"), "a") as f:
        f.write(
            "Epoch {}, Training loss: {:.6f}, Testing Acc: {:.4f}\n".format(
                epoch, avg_train_loss, avg_test_acc
            )
        )

writer.close()
