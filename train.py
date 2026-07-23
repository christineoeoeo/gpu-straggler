import argparse
import json
import socket
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


DATASET_ROOT = "/tmp/kgzh0394/cifar10"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 worker training"
    )

    parser.add_argument(
        "--worker-id",
        type=str,
        default=socket.gethostname(),
        help="Worker name, for example PC1 or PC2",
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=12500,
        help="Size of the worker's fixed local dataset",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of local mini-batches processed during "
            "the complete training run. By default, all batches are processed."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Artificial delay added after every batch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def create_dataset(subset_size, seed):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    full_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        download=False,
        transform=transform,
    )

    if subset_size <= 0:
        raise ValueError("--subset-size must be greater than zero")

    if subset_size > len(full_dataset):
        raise ValueError(
            f"Requested {subset_size} images, but CIFAR-10 "
            f"contains only {len(full_dataset)} training images"
        )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(full_dataset),
        generator=generator,
    )[:subset_size]

    dataset = Subset(
        full_dataset,
        indices.tolist(),
    )

    return dataset


def main():
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected")

    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be greater than zero")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")
    hostname = socket.gethostname()

    dataset = create_dataset(
        subset_size=args.subset_size,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = resnet18(
        weights=None,
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
    )

    print(f"Worker: {args.worker_id}")
    print(f"Hostname: {hostname}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Local dataset size: {len(loader.dataset)}")
    print(f"Maximum batches: {args.max_batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"CPU DataLoader workers: {args.num_workers}")
    print(f"Artificial delay: {args.delay}s")

    total_correct = 0
    total_samples_processed = 0
    total_batches_processed = 0

    iteration_times = []
    epoch_times = []

    model.train()

    stop_training = False

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        epoch_batches_processed = 0

        for batch_index, (images, labels) in enumerate(loader):
            # Stop when this worker reaches its assigned work budget.
            if (
                args.max_batches is not None
                and total_batches_processed >= args.max_batches
            ):
                stop_training = True
                break

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

            if args.delay > 0:
                time.sleep(args.delay)

            torch.cuda.synchronize()

            iteration_time = (
                time.perf_counter() - iteration_start
            )

            batch_samples = labels.size(0)

            batch_throughput = (
                batch_samples / iteration_time
                if iteration_time > 0
                else 0.0
            )

            iteration_times.append(iteration_time)

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples_processed += batch_samples
            total_batches_processed += 1
            epoch_batches_processed += 1

            progress_metrics = {
                "type": "progress",
                "worker_id": args.worker_id,
                "hostname": hostname,
                "epoch": epoch + 1,
                "batch": batch_index,
                "total_batches_processed": total_batches_processed,
                "iteration_time": round(
                    iteration_time,
                    6,
                ),
                "throughput": round(
                    batch_throughput,
                    2,
                ),
                "loss": round(
                    loss.item(),
                    6,
                ),
            }

            print(
                "METRIC "
                + json.dumps(progress_metrics),
                flush=True,
            )

        torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        # Only count epochs in which work was processed.
        if epoch_batches_processed > 0:
            epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"{epoch_time:.2f}s, "
            f"{epoch_batches_processed} batches processed"
        )

        if stop_training:
            break

    total_training_time = sum(epoch_times)

    average_iteration_time = (
        sum(iteration_times) / len(iteration_times)
        if iteration_times
        else 0.0
    )

    overall_throughput = (
        total_samples_processed / total_training_time
        if total_training_time > 0
        else 0.0
    )

    training_accuracy = (
        100.0 * total_correct / total_samples_processed
        if total_samples_processed > 0
        else 0.0
    )

    final_metrics = {
        "type": "final",
        "worker_id": args.worker_id,
        "hostname": hostname,
        "gpu": torch.cuda.get_device_name(0),
        "epochs_requested": args.epochs,
        "local_dataset_size": len(loader.dataset),
        "max_batches": args.max_batches,
        "batches_processed": total_batches_processed,
        "samples_processed": total_samples_processed,
        "epoch_time": round(
            total_training_time,
            6,
        ),
        "average_iteration_time": round(
            average_iteration_time,
            6,
        ),
        "throughput": round(
            overall_throughput,
            2,
        ),
        "training_accuracy": round(
            training_accuracy,
            2,
        ),
        "delay": args.delay,
    }

    print(
        "FINAL_METRICS "
        + json.dumps(final_metrics),
        flush=True,
    )


if __name__ == "__main__":
    main()
