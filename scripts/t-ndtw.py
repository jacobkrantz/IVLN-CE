import argparse
import json
import time

from habitat_extensions.tour_ndtw import compute_tour_ndtw


def main():
    parser = argparse.ArgumentParser(description="Compute tour nDTW.")
    parser.add_argument(
        "--gt-path", default="data/gt_ndtw.json", type=str, required=False
    )
    parser.add_argument(
        "--agent-path", default="agent_path.json", type=str, required=False
    )
    parser.add_argument(
        "--success-distance", default=3.0, type=float, required=False
    )
    parser.add_argument(
        "--split", default="val_unseen", type=str, required=False
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = time.time()

    with open(args.agent_path, "r") as f:
        agent_path = json.load(f)

    with open(args.gt_path, "r") as f:
        gt_path = json.load(f)[args.split]

    start2 = time.time()
    tour_ndtw = compute_tour_ndtw(
        agent_path, gt_path, args.success_distance, args.verbose
    )
    print(f"t-ndtw: {100 * tour_ndtw}")
    print(
        "script time:",
        round(time.time() - start, 1),
        "Alg time:",
        round(time.time() - start2, 1),
    )


if __name__ == "__main__":
    main()
