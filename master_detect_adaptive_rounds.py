import concurrent.futures
import csv
import json
import statistics
import subprocess
import time


USERNAME = "kgzh0394"
workers = {
    "PC1": "localhost",
    "PC2": "129.11.144.79",
    "PC3": "129.11.144.252",
    "PC4": "129.11.146.207",
}

project_dir = "/tmp/kgzh0394/gpu_straggler_project"
python_path = "/tmp/kgzh0394/ddnn-env/bin/python"

TOTAL_IMAGES = 2000
NUM_ROUNDS = 30

# Round 1 starts with equal allocation
allocation = {
    "PC1": 500,
    "PC2": 500,
    "PC3": 500,
    "PC4": 500,
}

# Straggler threshold = 85% of median throughput
THRESHOLD_RATIO = 0.85

RESULTS_FILE = "adaptive_round_results.csv"
SUMMARY_FILE = "adaptive_round_summary.csv"


def run_worker(worker_id, host, subset_size):

    command = (
        f"cd {project_dir} && "
        f"{python_path} train.py "
        f"--worker-id {worker_id} "
        f"--subset-size {subset_size}"
    )

    # PC2 is the controlled straggler
    if worker_id == "PC2":
        command = (
            f"cd {project_dir} && "
            f"taskset -c 0 {python_path} train.py "
            f"--worker-id {worker_id} "
            f"--subset-size {subset_size}"
        )

    if host == "localhost":
        final_command = ["bash", "-lc", command]
    else:
        final_command = [
            "ssh",
            f"{USERNAME}@{host}",
            command,
        ]

    print(
        f"Starting {worker_id} "
        f"with {subset_size} images..."
    )

    result = subprocess.run(
        final_command,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{worker_id} failed:\n{result.stderr}"
        )

    for line in result.stdout.splitlines():

        if line.startswith("FINAL_METRICS "):

            metrics = json.loads(
                line.removeprefix("FINAL_METRICS ")
            )

            metrics["allocated_images"] = subset_size

            return metrics

    raise RuntimeError(
        f"No FINAL_METRICS received from {worker_id}"
    )


def run_workers(current_allocation):

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        futures = {
            executor.submit(
                run_worker,
                worker_id,
                host,
                current_allocation[worker_id],
            ): worker_id
            for worker_id, host in workers.items()
        }

        for future in concurrent.futures.as_completed(futures):

            worker_id = futures[future]

            try:
                results.append(future.result())
                print(f"Completed {worker_id}")

            except Exception as error:
                print(
                    f"ERROR {worker_id}: {error}"
                )

    if len(results) != len(workers):
        raise RuntimeError(
            "Not all workers returned results."
        )

    return results


def detect_stragglers(results):

    start = time.perf_counter()

    throughputs = [
        result["throughput"]
        for result in results
    ]

    median_throughput = statistics.median(
        throughputs
    )

    threshold = (
        median_throughput
        * THRESHOLD_RATIO
    )

    stragglers = [
        result["worker_id"]
        for result in results
        if result["throughput"] < threshold
    ]

    runtime = time.perf_counter() - start

    return (
        stragglers,
        median_throughput,
        threshold,
        runtime,
    )


def calculate_adaptive_allocation(results):

    start = time.perf_counter()

    speeds = {
        result["worker_id"]:
        result["throughput"]
        for result in results
    }

    total_speed = sum(
        speeds.values()
    )

    if total_speed <= 0:
        raise ValueError(
            "Invalid throughput values."
        )

    raw = {
        worker_id:
        (
            speed
            / total_speed
            * TOTAL_IMAGES
        )
        for worker_id, speed
        in speeds.items()
    }

    new_allocation = {
        worker_id: int(value)
        for worker_id, value in raw.items()
    }

    remaining = (
        TOTAL_IMAGES
        - sum(new_allocation.values())
    )

    # Distribute rounding remainder
    # to workers with largest fractions
    order = sorted(
        raw,
        key=lambda worker_id:
        raw[worker_id]
        - new_allocation[worker_id],
        reverse=True,
    )

    for worker_id in order[:remaining]:
        new_allocation[worker_id] += 1

    runtime = time.perf_counter() - start

    return new_allocation, runtime


def allocation_change(old, new):

    return sum(
        abs(
            new[worker_id]
            - old[worker_id]
        )
        for worker_id in workers
    )


def save_results(
    round_number,
    allocation_now,
    results,
    median,
    threshold,
    stragglers,
    change,
    completion_time,
    detection_runtime,
    allocation_runtime,
    algorithm_runtime,
):

    throughput = {
        result["worker_id"]:
        result["throughput"]
        for result in results
    }

    row = {
        "round": round_number,

        "PC1_allocation":
            allocation_now["PC1"],

        "PC2_allocation":
            allocation_now["PC2"],

        "PC3_allocation":
            allocation_now["PC3"],

        "PC4_allocation":
            allocation_now["PC4"],

        "PC1_throughput":
            throughput["PC1"],

        "PC2_throughput":
            throughput["PC2"],

        "PC3_throughput":
            throughput["PC3"],

        "PC4_throughput":
            throughput["PC4"],

        "median_throughput":
            median,

        "threshold":
            threshold,

        "stragglers":
            ",".join(stragglers),

        "allocation_change":
            change,

        "round_completion_time_s":
            completion_time,

        "detection_runtime_s":
            detection_runtime,

        "allocation_runtime_s":
            allocation_runtime,

        "algorithm_runtime_s":
            algorithm_runtime,
    }

    file_exists = False

    try:
        with open(
            RESULTS_FILE,
            "r",
            newline=""
        ):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(
        RESULTS_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():

    global allocation

    print("=" * 60)
    print("ADAPTIVE GPU SCHEDULING")
    print("=" * 60)

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        print("\n" + "=" * 60)
        print(
            f"ROUND {round_number}"
        )
        print("=" * 60)

        print("\nCurrent allocation:")

        for worker_id in workers:

            print(
                f"{worker_id}: "
                f"{allocation[worker_id]} images"
            )

        # ------------------------------------------------
        # 1. Run workers
        # ------------------------------------------------

        results = run_workers(
            allocation
        )

        # ------------------------------------------------
        # 2. Start scheduling decision timer
        # ------------------------------------------------

        algorithm_start = (
            time.perf_counter()
        )

        # ------------------------------------------------
        # 3. Detect stragglers
        # ------------------------------------------------

        (
            stragglers,
            median,
            threshold,
            detection_runtime,
        ) = detect_stragglers(
            results
        )

        # ------------------------------------------------
        # 4. Calculate adaptive allocation
        # ------------------------------------------------

        new_allocation, allocation_runtime = (
            calculate_adaptive_allocation(
                results
            )
        )

        algorithm_runtime = (
            time.perf_counter()
            - algorithm_start
        )

        # ------------------------------------------------
        # 5. Calculate round completion time
        # ------------------------------------------------

        epoch_times = [
            result["epoch_time"]
            for result in results
        ]

        completion_time = max(
            epoch_times
        )

        # ------------------------------------------------
        # 6. Calculate allocation change
        # ------------------------------------------------

        change = allocation_change(
            allocation,
            new_allocation
        )

        # ------------------------------------------------
        # Display results
        # ------------------------------------------------

        print("\nResults:")

        for result in sorted(
            results,
            key=lambda x:
            x["worker_id"]
        ):

            print(
                f"{result['worker_id']}: "
                f"{result['allocated_images']} images | "
                f"{result['throughput']:.2f} images/s | "
                f"{result['epoch_time']:.3f}s"
            )

        print(
            f"\nMedian throughput: "
            f"{median:.2f}"
        )

        print(
            f"Detection threshold: "
            f"{threshold:.2f}"
        )

        print("\nDetection:")

        for worker_id in workers:

            status = (
                "STRAGGLER"
                if worker_id in stragglers
                else "NORMAL"
            )

            print(
                f"{worker_id}: {status}"
            )

        print("\nNew adaptive allocation:")

        for worker_id in workers:

            percentage = (
                new_allocation[worker_id]
                / TOTAL_IMAGES
                * 100
            )

            print(
                f"{worker_id}: "
                f"{new_allocation[worker_id]} images "
                f"({percentage:.2f}%)"
            )

        print(
            f"\nRound completion time: "
            f"{completion_time:.3f}s"
        )

        print(
            f"Allocation change: "
            f"{change} images"
        )

        print(
            f"Algorithm runtime: "
            f"{algorithm_runtime * 1e6:.2f} μs"
        )

        # ------------------------------------------------
        # Save current round
        # ------------------------------------------------

        save_results(
            round_number,
            allocation,
            results,
            median,
            threshold,
            stragglers,
            change,
            completion_time,
            detection_runtime,
            allocation_runtime,
            algorithm_runtime,
        )

        # ------------------------------------------------
        # New allocation becomes next round allocation :))
        # ------------------------------------------------

        allocation = new_allocation

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
