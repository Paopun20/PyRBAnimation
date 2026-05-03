import json, math, argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def normalize_q(q):
    q = np.array(q, dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def smoothstep(t):
    return t * t * (3 - 2 * t)


# ─────────────────────────────────────────────
# CFrame helpers (Roblox safe)
# ─────────────────────────────────────────────

def parse_cframe(cf):
    cf = np.array(cf, dtype=np.float32)
    if len(cf) != 12:
        return None
    return cf.reshape(3, 4)


# ─────────────────────────────────────────────
# Pose extraction
# ─────────────────────────────────────────────

def extract_pose(pose):
    pos = np.array(pose["Position"], dtype=np.float32)
    rot = np.array(pose["Rotation"], dtype=np.float32)
    return pos, rot


# ─────────────────────────────────────────────
# Smoothing (velocity-based)
# ─────────────────────────────────────────────

def smooth_velocity(x, alpha=0.85):
    x = np.array(x, dtype=np.float32)

    v = np.diff(x, axis=0, prepend=x[0:1])
    v = gaussian_filter1d(v, sigma=1.0, axis=0)

    out = np.cumsum(v * alpha, axis=0)
    out += x[0]

    return out.astype(np.float32)


# ─────────────────────────────────────────────
# Quaternion SLERP (fixed)
# ─────────────────────────────────────────────

def slerp(q0, q1, t):
    q0 = normalize_q(q0)
    q1 = normalize_q(q1)

    dot = np.dot(q0, q1)

    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return normalize_q(q0 + t * (q1 - q0))

    theta = np.arccos(dot)
    sin_t = np.sin(theta)

    a = np.sin((1 - t) * theta) / sin_t
    b = np.sin(t * theta) / sin_t

    return normalize_q(a * q0 + b * q1)


# ─────────────────────────────────────────────
# Time remap
# ─────────────────────────────────────────────

def remap_time(n):
    t = np.linspace(0, 1, n)
    return smoothstep(t)


# ─────────────────────────────────────────────
# Core smoothing pipeline
# ─────────────────────────────────────────────

def process_animation(data, alpha=0.85):
    times = []
    positions = []
    rotations = []

    # ── extract ─────────────────────────────
    for kf in data:
        times.append(kf["Time"])

        # take first pose only (safe minimal assumption)
        if len(kf["Poses"]) > 0:
            pos, rot = extract_pose(kf["Poses"][0])
        else:
            pos = np.zeros(3, dtype=np.float32)
            rot = np.zeros(3, dtype=np.float32)

        positions.append(pos)
        rotations.append(rot)

    positions = np.array(positions, dtype=np.float32)
    rotations = np.array(rotations, dtype=np.float32)

    # ── smoothing ───────────────────────────
    positions = smooth_velocity(positions, alpha=alpha)

    # ── time easing ─────────────────────────
    eased = remap_time(len(times))

    return times, positions, rotations, eased


# ─────────────────────────────────────────────
# Load / Save
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("Loading:", args.input)
    data = load_json(args.input)

    t, pos, rot, eased = process_animation(data, args.alpha)

    if args.debug:
        print("Frames:", len(t))
        print("Position sample:", pos[:2])
        print("Rotation sample:", rot[:2])
        print("Easing sample:", eased[:5])

    # ── rebuild minimal output ─────────────────
    out = []

    for i in range(len(t)):
        out.append({
            "Time": float(t[i]),
            "Position": pos[i].tolist(),
            "Rotation": rot[i].tolist(),
            "Ease": float(eased[i])
        })

    save_json(out, args.output)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
