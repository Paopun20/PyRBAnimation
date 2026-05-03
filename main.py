import json, math, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


# ── CFrame <-> quaternion ────────────────────────────────────────────────────


def mat_to_quat(r):
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def quat_to_mat(qw, qx, qy, qz):
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def cframe_to_vec(c):
    r = np.array(c[3:12]).reshape(3, 3)
    qw, qx, qy, qz = mat_to_quat(r)
    return np.array([c[0], c[1], c[2], qx, qy, qz, qw], dtype=np.float32)


def vec_to_cframe(v):
    q = v[3:7]
    q = q / max(np.linalg.norm(q), 1e-8)
    r = quat_to_mat(q[3], q[0], q[1], q[2])
    return [float(v[0]), float(v[1]), float(v[2])] + r.flatten().tolist()


def renorm(tensor):
    norms = np.linalg.norm(tensor[..., 3:7], axis=-1, keepdims=True)
    tensor[..., 3:7] /= np.maximum(norms, 1e-8)
    return tensor


# ── JSON <-> tensor ──────────────────────────────────────────────────────────


def load_anim(path):
    return json.load(open(path))


def save_anim(data, path):
    json.dump(data, open(path, "w"), indent=2)


def collect_bones(data):
    names = set()

    def walk(poses):
        for p in poses:
            names.add(p["Name"])
            walk(p.get("Children", []))

    for kf in data:
        walk(kf["Poses"])
    return sorted(names)


ID7 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)


def anim_to_tensor(data, bones):
    bi = {n: i for i, n in enumerate(bones)}
    T, B = len(data), len(bones)
    times = np.array([kf["Time"] for kf in data], dtype=np.float32)
    tensor = np.tile(ID7, (T, B, 1)).astype(np.float32)

    def walk(poses, t):
        for p in poses:
            if p["Name"] in bi:
                tensor[t, bi[p["Name"]]] = cframe_to_vec(p["CFrame"])
            walk(p.get("Children", []), t)

    for t, kf in enumerate(data):
        walk(kf["Poses"], t)
    return times, tensor


def tensor_to_anim(times, tensor, template, bones):
    bi = {n: i for i, n in enumerate(bones)}
    tensor = renorm(tensor.copy())

    def rebuild(poses, t):
        out = []
        for p in poses:
            np2 = copy.deepcopy(p)
            if p["Name"] in bi:
                np2["CFrame"] = vec_to_cframe(tensor[t, bi[p["Name"]]])
            np2["Children"] = rebuild(p.get("Children", []), t)
            out.append(np2)
        return out

    return [
        {"Name": kf["Name"], "Time": float(times[t]), "Poses": rebuild(kf["Poses"], t)}
        for t, kf in enumerate(template)
    ]


def synth_template(data, times):
    base = data[0]
    return [
        {
            "Name": f"Keyframe{i}",
            "Time": float(t),
            "Poses": copy.deepcopy(base["Poses"]),
        }
        for i, t in enumerate(times)
    ]


# ── Transforms ───────────────────────────────────────────────────────────────


def spline_upsample(times, tensor, factor):
    T, B, D = tensor.shape
    if T < 2:
        return times, tensor

    # Sort by time
    order = np.argsort(times, kind="stable")
    times, tensor = times[order], tensor[order]

    # Remove duplicate timestamps (keep last occurrence per time value)
    _, unique_idx = np.unique(times, return_index=True)
    if len(unique_idx) < len(times):
        removed = len(times) - len(unique_idx)
        print(f"  [warn] removed {removed} duplicate-time keyframe(s)")
        times, tensor = times[unique_idx], tensor[unique_idx]
        T = len(times)

    if T < 2:
        return times, tensor

    # CubicSpline needs >=4 points; fall back to linear for short sequences
    kind = "cubic" if T >= 4 else "linear"
    nt = np.linspace(times[0], times[-1], (T - 1) * factor + 1).astype(np.float32)
    out = np.zeros((len(nt), B, D), dtype=np.float32)
    for b in range(B):
        for d in range(D):
            if kind == "cubic":
                # clamped = zero first-derivative at endpoints, prevents start/end overshoot
                out[:, b, d] = CubicSpline(times, tensor[:, b, d], bc_type="clamped")(nt)
            else:
                out[:, b, d] = np.interp(nt, times, tensor[:, b, d])
    return nt, renorm(out)


def gauss_smooth(tensor, sigma):
    # mode='nearest' clamps boundary values, no ripple at frame 0
    return renorm(
        gaussian_filter1d(tensor, sigma=sigma, axis=0, mode="nearest").astype(np.float32)
    )


def jitter(tensor, ps=0.05, rs=0.025):
    """IID per-frame noise — GAN training only."""
    out = tensor.copy()
    out[:, :, 0:3] += (np.random.randn(*out[:, :, 0:3].shape) * ps).astype(np.float32)
    out[:, :, 3:7] += (np.random.randn(*out[:, :, 3:7].shape) * rs).astype(np.float32)
    return renorm(out)


# ── GAN ──────────────────────────────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(self, F, h=128, L=2):
        super().__init__()
        self.rnn = nn.GRU(
            F, h, L, batch_first=True, bidirectional=True, dropout=0.1 if L > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.LeakyReLU(0.2), nn.Linear(h, F)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h)


class Discriminator(nn.Module):
    def __init__(self, F, h=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(F, h, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(h, h * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(h * 2, 1)

    def forward(self, x):
        return self.fc(self.conv(x.permute(0, 2, 1)).squeeze(-1))


def train_gan(target, epochs, lr=1e-3, log_every=200):
    """
    target = the 'clean/smooth' reference tensor (T,B,D).
    Generator input  = jittered version of target.
    Generator output = should recover target.
    rec_loss weight  = 1.0  (NOT 10 — that caused zero-diff output).
    """
    T, B, D = target.shape
    F = B * D
    real = torch.tensor(target.reshape(1, T, F), dtype=torch.float32)
    G = Generator(F)
    Disc = Discriminator(F)
    oG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    oD = optim.Adam(Disc.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ones = torch.ones(1, 1)
    zeros = torch.zeros(1, 1)

    for ep in range(1, epochs + 1):
        noisy = torch.tensor(jitter(target).reshape(1, T, F), dtype=torch.float32)
        # D step
        oD.zero_grad()
        fake = G(noisy).detach()
        (bce(Disc(real), ones) + bce(Disc(fake), zeros)).backward()
        oD.step()
        # G step  — rec weight 1.0 so generator actually changes output
        oG.zero_grad()
        fake = G(noisy)
        loss = bce(Disc(fake), ones) + 1.0 * mse(fake, real)
        loss.backward()
        oG.step()
        if ep % log_every == 0 or ep == 1:
            with torch.no_grad():
                delta = (G(real) - real).abs().mean().item()
            print(f"    ep {ep:>5}/{epochs}  loss={loss.item():.4f}  Δ={delta:.5f}")
    return G


def apply_G(G, tensor):
    T, B, D = tensor.shape
    inp = torch.tensor(tensor.reshape(1, T, B * D), dtype=torch.float32)
    with torch.no_grad():
        out = G(inp).numpy().reshape(T, B, D)
    return renorm(out.astype(np.float32))


# ── Pipeline ─────────────────────────────────────────────────────────────────


def smooth_pipeline(times, tensor, upsample, sigma, use_gan, epochs):
    print(f"  spline upsample ×{upsample}")
    times, tensor = spline_upsample(times, tensor, upsample)
    print(f"  gaussian smooth σ={sigma}")
    tensor = gauss_smooth(tensor, sigma)
    if use_gan:
        print(f"  GAN ({epochs} epochs) — refines smooth manifold")
        G = train_gan(tensor, epochs)
        # small jitter so G has something to denoise — output ≠ input
        tensor = apply_G(G, jitter(tensor, ps=0.008, rs=0.004))
    return times, tensor


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--upsample", type=int, default=3, help="Keyframe multiplier (default 3)"
    )
    ap.add_argument(
        "--sigma", type=float, default=1.5, help="Gaussian sigma (default 1.5)"
    )
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument(
        "--no-gan", action="store_true", help="Skip GAN, spline/gaussian only"
    )
    args = ap.parse_args()

    print(f"\n── Loading {args.input}")
    data = load_anim(args.input)
    bones = collect_bones(data)
    times, tensor = anim_to_tensor(data, bones)
    orig_T = len(times)
    print(
        f"   {orig_T} keyframes  |  {len(bones)} bones  |  {times[0]:.3f}s–{times[-1]:.3f}s"
    )

    saved_times, saved_tensor = times.copy(), tensor.copy()

    print(f"\n── Smoothing  (GAN={'off' if args.no_gan else 'on'})")
    times, tensor = smooth_pipeline(
        times, tensor, args.upsample, args.sigma, not args.no_gan, args.epochs
    )

    print(f"\n── Diff report")
    print(f"   keyframes: {orig_T} → {len(times)}")
    if len(times) == len(saved_times):
        d = np.abs(tensor - saved_tensor)
        print(
            f"   position Δ  mean={d[:, :, 0:3].mean():.5f}  max={d[:, :, 0:3].max():.5f}"
        )
        print(
            f"   rotation Δ  mean={d[:, :, 3:7].mean():.5f}  max={d[:, :, 3:7].max():.5f}"
        )
    else:
        print(f"   (new timeline — compare via Roblox import)")

    tmpl = synth_template(data, times) if len(times) != orig_T else data
    save_anim(tensor_to_anim(times, tensor, tmpl, bones), args.output)
    print(f"\n── Saved {len(times)} keyframes → {args.output}\n")


if __name__ == "__main__":
    main()
