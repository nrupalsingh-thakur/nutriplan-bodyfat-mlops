"""Send fake user requests to the running API, to demo monitoring.

  python scripts/simulate_traffic.py --mode normal    # users like the training data -> status OK
  python scripts/simulate_traffic.py --mode shifted   # heavier, older users          -> status DRIFT
"""
import argparse
import json
import random
import urllib.request


def person(mode: str) -> dict:
    if mode == "normal":
        age, height, weight = random.gauss(45, 12), random.gauss(178, 7), random.gauss(81, 12)
        waist = 92 + (weight - 81) * 0.8 + random.gauss(0, 4)
    else:  # population changed: older and much heavier users
        age, height, weight = random.gauss(62, 8), random.gauss(176, 7), random.gauss(108, 10)
        waist = 115 + random.gauss(0, 5)
    return {
        "age": int(min(max(age, 18), 90)),
        "weight_kg": round(min(max(weight, 45), 195), 1),
        "height_cm": round(min(max(height, 150), 205), 1),
        "neck_cm": round(38 + (weight - 81) * 0.1 + random.gauss(0, 1.5), 1),
        "waist_cm": round(min(max(waist, 65), 155), 1),
        "hip_cm": round(100 + (weight - 81) * 0.5 + random.gauss(0, 3), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    random.seed(1 if a.mode == "normal" else 2)
    for _ in range(a.n):
        req = urllib.request.Request(f"{a.url}/predict", data=json.dumps(person(a.mode)).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
    print(f"Sent {a.n} '{a.mode}' requests.")
    print(json.dumps(json.loads(urllib.request.urlopen(f"{a.url}/monitoring/drift").read()), indent=2))


if __name__ == "__main__":
    main()
