
import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--allocation",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Number of samples assigned to each rank. "
            "Example: --allocation 528 384 592 496"
        ),
    )

    parser.add_argument(
        "--batch-allocation",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Local batch size assigned to each rank. "
            "Example: --batch-allocation 66 48 74 62"
        ),
    )

    return parser.parse_args()


def build_model():
    model = resnet18(
        weights=None,
        num_classes=10,
    )

    # Adapt ResNet18 to CIFAR-10's 32x32 images.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    return model


def main():
    args = parse_args()

    # torchrun provides these values.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.allocation is not None:
        if len(args.allocation) != world_size:
            raise ValueError(
                "--allocation must contain exactly "
                f"{world_size} values, but received "
                f"{len(args.allocation)}."
            )

        if any(value <= 0 for value in args.allocation):
            raise ValueError(
                "Every allocation value must be greater than zero."
            )

        local_sample_count = args.allocation[rank]
    else:
        local_sample_count = None

    if args.batch_allocation is not None:
        if len(args.batch_allocation) != world_size:
            raise ValueError(
                "--batch-allocation must contain exactly "
                f"{world_size} values, but received "
                f"{len(args.batch_allocation)}."
            )

        if any(value <= 0 for value in args.batch_allocation):
            raise ValueError(
                "Every batch-allocation value must be greater "
                "than zero."
            )

        local_batch_size = args.batch_allocation[rank]
    else:
        local_batch_size = args.batch_size

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    hostname = socket.gethostname()

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    dataset_root = f"/tmp/{os.environ['USER']}/cifar10"

    dataset = datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=False,
        transform=transform,
    )

    if args.allocation is not None:
        total_requested = sum(args.allocation)

        if total_requested > len(dataset):
            raise ValueError(
                "Total allocation exceeds dataset size. "
                f"Requested {total_requested}, "
                f"dataset contains {len(dataset)} samples."
            )

        start_index = sum(args.allocation[:rank])
        end_index = start_index + local_sample_count

        local_indices = list(range(start_index, end_index))
        local_dataset = Subset(dataset, local_indices)

        sampler = None

    else:
        local_dataset = dataset

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    loader = DataLoader(
        local_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    print(
        f"Rank {rank}/{world_size} | "
        f"Host {hostname} | "
        f"GPU {torch.cuda.get_device_name(local_rank)} | "
        f"Requested allocation={local_sample_count} | "
        f"Local dataset size={len(local_dataset)} | "
        f"Local batch size={local_batch_size} | "
        f"Batches={len(loader)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()

        epoch_correct = 0
        epoch_samples = 0
        iteration_times = []

        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # Allows ranks with fewer batches to finish without
        # causing the remaining DDP ranks to hang.
        with model.join():
            for batch_index, (images, labels) in enumerate(loader):
                torch.cuda.synchronize()
                iteration_start = time.perf_counter()

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()

                iteration_time = (
                    time.perf_counter() - iteration_start
                )

                iteration_times.append(iteration_time)

                batch_samples = labels.size(0)
                epoch_samples += batch_samples

                epoch_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                if batch_index % 25 == 0:
                    throughput = (
                        batch_samples / iteration_time
                    )

                    print(
                        f"METRIC "
                        f"rank={rank} "
                        f"epoch={epoch} "
                        f"batch={batch_index} "
                        f"configured_batch_size="
                        f"{local_batch_size} "
                        f"actual_batch_size="
                        f"{batch_samples} "
                        f"iteration_time="
                        f"{iteration_time:.6f} "
                        f"throughput="
                        f"{throughput:.2f}",
                        flush=True,
                    )

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        if iteration_times:
            average_iteration_time = (
                sum(iteration_times) / len(iteration_times)
            )
        else:
            average_iteration_time = 0.0

        if epoch_samples > 0:
            training_accuracy = (
                epoch_correct / epoch_samples * 100.0
            )

            epoch_throughput = (
                epoch_samples / epoch_time
            )
        else:
            training_accuracy = 0.0
            epoch_throughput = 0.0

        local_metrics = {
            "rank": rank,
            "hostname": hostname,
            "epoch": epoch,
            "epoch_time": epoch_time,
            "average_iteration_time": average_iteration_time,
            "throughput": epoch_throughput,
            "samples_processed": epoch_samples,
            "training_accuracy": training_accuracy,
            "requested_allocation": local_sample_count,
            "local_batch_size": local_batch_size,
            "batches_processed": len(iteration_times),
        }

        gathered_metrics = [
            None for _ in range(world_size)
        ]

        dist.all_gather_object(
            gathered_metrics,
            local_metrics,
        )

        if rank == 0:
            print("\nDDP RESULTS", flush=True)

            for metrics in gathered_metrics:
                print(
                    f"Rank {metrics['rank']} | "
                    f"{metrics['hostname']} | "
                    f"requested="
                    f"{metrics['requested_allocation']} | "
                    f"processed="
                    f"{metrics['samples_processed']} | "
                    f"batch_size="
                    f"{metrics['local_batch_size']} | "
                    f"batches="
                    f"{metrics['batches_processed']} | "
                    f"epoch="
                    f"{metrics['epoch_time']:.3f}s | "
                    f"iteration="
                    f"{metrics['average_iteration_time']:.6f}s | "
                    f"throughput="
                    f"{metrics['throughput']:.2f} | "
                    f"accuracy="
                    f"{metrics['training_accuracy']:.2f}%",
                    flush=True,
                )

            round_completion_time = max(
                metrics["epoch_time"]
                for metrics in gathered_metrics
            )

            print(
                f"ROUND_COMPLETION_TIME "
                f"{round_completion_time:.3f}s",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

