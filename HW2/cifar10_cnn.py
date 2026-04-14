# Don't forget to export HF_ENDPOINT = hf-mirror.com

import csv
import os
from typing import Optional, Tuple

from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import concatenate_datasets, load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

#! Data Part.
# Code agent is used during this session.
# CINIC-10 dataset as extended pretraining data.

# Reused from basic_cnn.py.
val_ratio = 0.1
batch_size = 1024
pretrain_learning_rate = 3e-4
finetune_learning_rate = 1e-3
pretrain_epoch = 80
finetune_epoch = 20
finetune_warmup_epochs = 0
weight_decay = 1e-3

DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))
LOG_POSTFIX_INTERVAL = int(os.environ.get("LOG_POSTFIX_INTERVAL", "50"))
# Whether to cold start the finetune optimizer.
FINETUNE_COLD_START = os.environ.get("FINETUNE_COLD_START", "0") == "1"
CINIC10_HF_DATASET = "flwrlabs/cinic10"


def load_cinic10_from_huggingface(
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)

    if num_proc is not None:
        return load_dataset(
            CINIC10_HF_DATASET, cache_dir=cache_dir, num_proc=num_proc
        )
    return load_dataset(CINIC10_HF_DATASET, cache_dir=cache_dir)


class Cinic10HFDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self._data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        image = row["image"]
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def setup_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1


def dataloader_kwargs(pin_memory: bool) -> dict:
    kw = {
        "num_workers": DATALOADER_NUM_WORKERS,
        "pin_memory": pin_memory,
    }
    if DATALOADER_NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def all_reduce_sum3(
    loss_sum: torch.Tensor, correct: torch.Tensor, total: torch.Tensor, distributed: bool
) -> Tuple[float, float, float]:
    if distributed:
        stacked = torch.stack([loss_sum, correct, total])
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        loss_sum, correct, total = stacked[0], stacked[1], stacked[2]
    return loss_sum.item(), correct.item(), total.item()


#! Model Part.
# This model is adapted from Attention Residual(Kimi Team, arxiv2603.15031)
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        :param x [Batch, Channel]
        """
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms + self.eps))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class FullAttnResStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        stride = 2 if in_channels != out_channels else 1

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.transform_layers = nn.ModuleList([ResBlock(out_channels) for _ in range(num_layers)])
        self.queries = nn.ParameterList([nn.Parameter(torch.zeros(out_channels)) for _ in range(num_layers)])
        self.k_norm = RMSNorm(out_channels)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        history_v = [x]

        for l in range(self.num_layers):
            if l == 0:
                weighted_sum = x
            else:
                scores = []
                for v_i in history_v:

                    k_i = F.adaptive_avg_pool2d(v_i, 1).view(v_i.size(0), -1)
                    k_i_normed = self.k_norm(k_i)

                    score = torch.matmul(k_i_normed, self.queries[l]).unsqueeze(1)
                    scores.append(score)

                attn_weights = F.softmax(torch.cat(scores, dim=1), dim=1)

                weighted_sum = 0
                for i, v_i in enumerate(history_v):
                    w = attn_weights[:, i].view(-1, 1, 1, 1)
                    weighted_sum = weighted_sum + v_i * w

            f_l = self.transform_layers[l](weighted_sum)
            h_l = weighted_sum + f_l
            history_v.append(h_l)

        return history_v[-1]


class AttnResNet(nn.Module):
    def __init__(self, stage_configs, num_classes=10):
        """
        stage_configs: list of tuples -> [(in_c, out_c, depth), ...]
        """
        super().__init__()

        first_in_c = stage_configs[0][0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_in_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_in_c),
            nn.ReLU(inplace=True)
        )

        stages = []
        for in_c, out_c, depth in stage_configs:
            stages.append(FullAttnResStage(in_c, out_c, depth))
        self.stages = nn.Sequential(*stages)

        last_out_c = stage_configs[-1][1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_out_c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def finetune_lr_schedule(epoch_idx):
    if epoch_idx >= finetune_warmup_epochs:
        return finetune_learning_rate
    if finetune_warmup_epochs <= 1:
        return finetune_learning_rate
    t = epoch_idx / (finetune_warmup_epochs - 1)
    return pretrain_learning_rate + (finetune_learning_rate - pretrain_learning_rate) * t


def main():
    distributed, local_rank, rank, _world_size = setup_distributed()
    is_main = rank == 0
    pin_memory = torch.cuda.is_available()
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")

    dl_kw = dataloader_kwargs(pin_memory)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    split_generator = torch.Generator().manual_seed(42)
    train_full_aug = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_full_eval = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_val_test
    )
    n_full = len(train_full_aug)
    n_val = int(n_full * val_ratio)
    n_train = n_full - n_val
    perm = torch.randperm(n_full, generator=split_generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = Subset(train_full_aug, train_indices)
    val_dataset = Subset(train_full_eval, val_indices)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val_test
    )

    cinic_raw = load_cinic10_from_huggingface()
    cinic_all = concatenate_datasets(
        [cinic_raw["train"], cinic_raw["validation"], cinic_raw["test"]]
    )
    cinic = Cinic10HFDataset(cinic_all, transform=transform_train)

    pretrain_dataset = ConcatDataset([cinic, train_dataset])

    if distributed:
        pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            sampler=pretrain_sampler,
            **dl_kw,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, **dl_kw
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler, **dl_kw
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=test_sampler, **dl_kw
        )
    else:
        pretrain_sampler = None
        train_sampler = None
        val_sampler = None
        test_sampler = None
        pretrain_loader = DataLoader(
            pretrain_dataset, batch_size=batch_size, shuffle=True, **dl_kw
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **dl_kw
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **dl_kw
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, **dl_kw
        )

    # Kind of Resnet
    model = AttnResNet(
        stage_configs=[(64, 64, 5), (64, 128, 5), (128, 128, 5)],
        num_classes=10,
    )
    model = model.to(device)
    if distributed:
        # FullAttnResStage in l==0 does not use queries[l], some parameters have no gradient in this step, so need to enable unused detection.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    criterion = nn.CrossEntropyLoss()
    pretrain_optimizer = torch.optim.Adam(
        model.parameters(), lr=pretrain_learning_rate, weight_decay=weight_decay
    )
    finetune_optimizer = torch.optim.Adam(
        model.parameters(), lr=finetune_learning_rate, weight_decay=weight_decay
    )

    # Pretrain phase.
    pretrain_log_path = os.path.join(os.path.dirname(__file__), "pretrain_log_config.csv")
    if is_main:
        pretrain_log_file = open(pretrain_log_path, "w", newline="")
        pretrain_csv = csv.writer(pretrain_log_file)
        pretrain_csv.writerow(["epoch", "train_loss", "train_acc"])
    else:
        pretrain_log_file = None

    try:
        for epoch in tqdm(
            range(pretrain_epoch), desc="Pretrain", disable=not is_main
        ):
            if pretrain_sampler is not None:
                pretrain_sampler.set_epoch(epoch)
            model.train()
            train_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
            train_correct = torch.zeros(1, device=device, dtype=torch.float64)
            train_total = torch.zeros(1, device=device, dtype=torch.float64)
            batch_pbar = tqdm(
                pretrain_loader,
                desc=f"pretrain ep {epoch + 1}/{pretrain_epoch}",
                leave=False,
                disable=not is_main,
            )
            for i, (images, labels) in enumerate(batch_pbar):
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                outputs = model(images)
                loss = criterion(outputs, labels)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

                batch_n = labels.size(0)
                with torch.no_grad():
                    train_loss_sum += (loss.detach().float() * batch_n).double()
                    train_correct += (outputs.argmax(1) == labels).sum().double()
                    train_total += batch_n

                if is_main and (i + 1) % LOG_POSTFIX_INTERVAL == 0:
                    ls = (train_loss_sum / train_total).item()
                    acc = (train_correct / train_total).item() * 100.0
                    batch_pbar.set_postfix(loss=f"{ls:.4f}", acc=f"{acc:.2f}%")

                if is_main and (i + 1) % 100 == 0:
                    ls = (train_loss_sum / train_total).item()
                    acc = (train_correct / train_total).item() * 100.0
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                            epoch + 1,
                            pretrain_epoch,
                            i + 1,
                            len(pretrain_loader),
                            ls,
                            acc,
                        )
                    )

            ls, cor, tot = all_reduce_sum3(
                train_loss_sum.squeeze(),
                train_correct.squeeze(),
                train_total.squeeze(),
                distributed,
            )
            epoch_train_loss = ls / tot
            epoch_train_acc = 100.0 * cor / tot
            if is_main:
                pretrain_csv.writerow([epoch + 1, epoch_train_loss, epoch_train_acc])
    finally:
        if pretrain_log_file is not None:
            pretrain_log_file.close()

    if distributed:
        dist.barrier()

    # Avoid Adam Problems.
    if not FINETUNE_COLD_START:
        finetune_optimizer.load_state_dict(pretrain_optimizer.state_dict())
        for g in finetune_optimizer.param_groups:
            g["lr"] = finetune_learning_rate

    # Finetune phase (train_dataset only); validation after each epoch.
    train_log_path = os.path.join(os.path.dirname(__file__), "train_log_config.csv")
    val_log_path = os.path.join(os.path.dirname(__file__), "val_log_config.csv")
    if is_main:
        train_log_file = open(train_log_path, "w", newline="")
        val_log_file = open(val_log_path, "w", newline="")
        train_csv = csv.writer(train_log_file)
        val_csv = csv.writer(val_log_file)
        train_csv.writerow(["epoch", "train_loss", "train_acc"])
        val_csv.writerow(["epoch", "val_loss", "val_acc"])
    else:
        train_log_file = None
        val_log_file = None

    try:
        for epoch in tqdm(
            range(finetune_epoch), desc="Finetune", disable=not is_main
        ):
            current_lr = finetune_lr_schedule(epoch)
            for g in finetune_optimizer.param_groups:
                g["lr"] = current_lr
            if is_main:
                print(
                    "Finetune epoch {}/{}  lr={:.6g}".format(
                        epoch + 1, finetune_epoch, current_lr
                    )
                )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
            train_correct = torch.zeros(1, device=device, dtype=torch.float64)
            train_total = torch.zeros(1, device=device, dtype=torch.float64)
            batch_pbar = tqdm(
                train_loader,
                desc=f"finetune train ep {epoch + 1}/{finetune_epoch}",
                leave=False,
                disable=not is_main,
            )
            for i, (images, labels) in enumerate(batch_pbar):
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                outputs = model(images)
                loss = criterion(outputs, labels)

                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()

                batch_n = labels.size(0)
                with torch.no_grad():
                    train_loss_sum += (loss.detach().float() * batch_n).double()
                    train_correct += (outputs.argmax(1) == labels).sum().double()
                    train_total += batch_n

                if is_main and (i + 1) % LOG_POSTFIX_INTERVAL == 0:
                    ls = (train_loss_sum / train_total).item()
                    acc = (train_correct / train_total).item() * 100.0
                    batch_pbar.set_postfix(loss=f"{ls:.4f}", acc=f"{acc:.2f}%")

                if is_main and (i + 1) % 100 == 0:
                    ls = (train_loss_sum / train_total).item()
                    acc = (train_correct / train_total).item() * 100.0
                    print(
                        "Finetune Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
                            epoch + 1,
                            finetune_epoch,
                            i + 1,
                            len(train_loader),
                            ls,
                            acc,
                        )
                    )

            ls, cor, tot = all_reduce_sum3(
                train_loss_sum.squeeze(),
                train_correct.squeeze(),
                train_total.squeeze(),
                distributed,
            )
            epoch_train_loss = ls / tot
            epoch_train_acc = 100.0 * cor / tot
            if is_main:
                train_csv.writerow([epoch + 1, epoch_train_loss, epoch_train_acc])

            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            model.eval()
            val_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
            val_correct = torch.zeros(1, device=device, dtype=torch.float64)
            val_total = torch.zeros(1, device=device, dtype=torch.float64)
            with torch.no_grad():
                for images, labels in tqdm(
                    val_loader,
                    desc=f"finetune val ep {epoch + 1}/{finetune_epoch}",
                    leave=False,
                    disable=not is_main,
                ):
                    images = images.to(device, non_blocking=pin_memory)
                    labels = labels.to(device, non_blocking=pin_memory)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    batch_n = labels.size(0)
                    val_loss_sum += (loss.detach().float() * batch_n).double()
                    val_correct += (outputs.argmax(1) == labels).sum().double()
                    val_total += batch_n

            ls, cor, tot = all_reduce_sum3(
                val_loss_sum.squeeze(),
                val_correct.squeeze(),
                val_total.squeeze(),
                distributed,
            )
            val_loss = ls / tot
            val_acc = 100.0 * cor / tot
            if is_main:
                val_csv.writerow([epoch + 1, val_loss, val_acc])
                print(
                    "Finetune Epoch [{}/{}] Val - Loss: {:.4f}, Accuracy: {:.2f}%".format(
                        epoch + 1, finetune_epoch, val_loss, val_acc
                    )
                )
    finally:
        if train_log_file is not None:
            train_log_file.close()
        if val_log_file is not None:
            val_log_file.close()

    if distributed:
        dist.barrier()

    # Final test on test set.
    test_log_path = os.path.join(os.path.dirname(__file__), "test_log_config.csv")
    model.eval()
    test_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    test_correct = torch.zeros(1, device=device, dtype=torch.float64)
    test_total = torch.zeros(1, device=device, dtype=torch.float64)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", disable=not is_main):
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_n = labels.size(0)
            test_loss_sum += (loss.detach().float() * batch_n).double()
            test_correct += (outputs.argmax(1) == labels).sum().double()
            test_total += batch_n

    ls, cor, tot = all_reduce_sum3(
        test_loss_sum.squeeze(),
        test_correct.squeeze(),
        test_total.squeeze(),
        distributed,
    )
    test_loss = ls / tot
    test_acc = 100.0 * cor / tot

    if is_main:
        with open(test_log_path, "w", newline="") as f:
            test_csv = csv.writer(f)
            test_csv.writerow(["test_loss", "test_acc"])
            test_csv.writerow([test_loss, test_acc])
        print("Test - Loss: {:.4f}, Accuracy: {:.2f}%".format(test_loss, test_acc))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
