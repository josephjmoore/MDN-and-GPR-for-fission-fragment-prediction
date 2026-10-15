import glob, re
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from pathlib import Path

def endf_float(s):#
    """
    Convert a string in ENDF-6 format to a float. 
    Handles empty strings and scientific notation without an 'e'.
    """
    s = s.strip()
    if not s: return 0.0
    i = max(s.rfind("+"), s.rfind("-"))
    if i > 0 and s[i-1] not in "eE":
        s = s[:i] + "e" + s[i:]
    return float(s)

def fields(line):
    return [line[k*11:(k+1)*11] for k in range(6)]

def read_nfpy(path, mt=454):
    """
    Reads a .dat file and extrats fission yield data.
    ZA reads target nuclide from Z*1000 + A encoding
    Blocks is a list (E, arr) where E is the incident energy and arr has four values per row: 
        -ZAFP - product nuclide identifier
        -FPS - product data field (not used later)
        -Y - yield
        -dY - yield uncertainty
    """
    with open(path) as fh:
        lines = [l.rstrip("\n").ljust(80) for l in fh]
    sec = [l for l in lines if l[70:72].strip() == "8" and l[72:75].strip() == str(mt)]
    ZA = int(round(endf_float(fields(sec[0])[0])))
    blocks, i = [], 1
    while i < len(sec):                      # one LIST record per energy
        f = fields(sec[i])
        E, NW, NFP = endf_float(f[0]), int(f[4]), int(f[5])
        n = -(-NW // 6)
        vals = [endf_float(x) for l in sec[i+1:i+1+n] for x in fields(l)][:NW]
        blocks.append((E, np.array(vals).reshape(NFP, 4)))
        i += 1 + n
    return ZA, blocks

def mass_yield(arr):
    """
    For a given array of fission product data, computes the mass yield distribution.
    decodes each row in array to give Z, A. Light nuclei filtered out (Z < 20).
    yA stores total yield for each A
    vA stores sum of squared uncertainties for each A
    Returns:
    - A: sorted array of mass numbers
    - Y: corresponding yields for each mass number
    - sY: corresponding propagated uncertainties for each mass number
    """
    yA, vA = defaultdict(float), defaultdict(float)
    for zafp, fps, y, dy in arr:
        z, a = divmod(int(round(zafp)), 1000)
        if z < 20: continue
        yA[a] += y; vA[a] += dy**2
    A = np.array(sorted(yA))
    return A, np.array([yA[a] for a in A]), np.sqrt([vA[a] for a in A])

# ---- S_n from AME2020 ----
def load_mass_excess(path="mass_1.mas20.txt", skip=36):
    """
    Returns (Z, A): mass excess in keV.#
    Used for calculating neutron separation energies S_n.
    """
    me = {}
    with open(path) as fh:
        for line in fh.read().splitlines()[skip:]:
            try:
                Z = int(line[9:14]); A = int(line[14:19])
                me[(Z, A)] = float(line[28:42].replace("#", "."))
            except ValueError:
                continue
    return me

ME = load_mass_excess(Path(__file__).resolve().parent / "mass_1.mas20.txt")
DELTA_N = 8071.317 # neutron mass excess in keV

def Sn_MeV(Z, A):
    """
    Returns neutron separation energy S_n in MeV for a given nuclide (Z, A).
    Used to estimate the excitation energy of the compound nucleus in neutron-induced fission.
    """
    return (ME[(Z, A-1)] + DELTA_N - ME[(Z, A)]) / 1000.0

base_dir = Path(__file__).resolve().parent
folder = base_dir / "neutron induced"
files = sorted(glob.glob(os.path.join(folder, "*.dat")))

rows = []   # Each row: Zc - compound charge number, Ac-Zc -compound neutron number, Ac -compound mass number, 
            #           Es- excitation energy in MeV, A_f - fragment mass number, Y - yield, sigY - uncertainty, 
            #           cid - file identifier
curve_records = []
for cid, path in enumerate(files):
    if re.search(r"\d+m_", path): continue           # skip isomeric targets
    ZA, blocks = read_nfpy(path)
    Zt, At = divmod(ZA, 1000)
    Zc, Ac = Zt, At + 1
    for E_eV, arr in blocks:
        E_incident = E_eV / 1e6
        Es = E_incident + Sn_MeV(Zc, Ac)
        A, Y, sY = mass_yield(arr)
        print(path, E_eV, round(Y.sum(), 4))        # should be ~2
        curve_records.append({
            "target_Z": Zc,
            "target_A": At,
            "incident_energy_eV": E_eV,
            "incident_energy_MeV": E_incident,
            "excitation_energy_MeV": Es,
            "fragment_A": A,
            "yields": Y,
            "yield_uncertainties": sY,
            "curve_id": len(curve_records),
            "file_id": cid,
            "source_file": Path(path).name,
        })
        for a, y, s in zip(A, Y, sY):
            rows.append([Zc, Ac-Zc, Ac, Es, a, y, s, cid])

data = np.array(rows)
np.save(base_dir / "endf_table.npy", data)
print(data.shape, "rows;", len(set(data[:, 7])), "files")

offsets = np.zeros(len(curve_records) + 1, dtype=np.int64)
for index, record in enumerate(curve_records):
    offsets[index + 1] = offsets[index] + len(record["fragment_A"])

np.savez_compressed(
    base_dir / "yield_curves.npz",
    target_Z=np.array([record["target_Z"] for record in curve_records], dtype=np.int64),
    target_A=np.array([record["target_A"] for record in curve_records], dtype=np.int64),
    incident_energy_eV=np.array(
        [record["incident_energy_eV"] for record in curve_records]
    ),
    incident_energy_MeV=np.array(
        [record["incident_energy_MeV"] for record in curve_records]
    ),
    excitation_energy_MeV=np.array(
        [record["excitation_energy_MeV"] for record in curve_records]
    ),
    curve_id=np.array([record["curve_id"] for record in curve_records], dtype=np.int64),
    file_id=np.array([record["file_id"] for record in curve_records], dtype=np.int64),
    source_file=np.array([record["source_file"] for record in curve_records]),
    offsets=offsets,
    fragment_A=np.concatenate([record["fragment_A"] for record in curve_records]),
    yields=np.concatenate([record["yields"] for record in curve_records]),
    yield_uncertainties=np.concatenate(
        [record["yield_uncertainties"] for record in curve_records]
    ),
)
print(f"Saved {len(curve_records)} curves to yield_curves.npz")

path = os.path.join(folder, "nfpy_092-U-235_9228.dat")
ZA, blocks = read_nfpy(path)

plt.figure(figsize=(10, 6))

for E_eV, arr in blocks:
    A, Y, sigmaY = mass_yield(arr)
    Ypct = 100 * Y / Y.sum()
    plt.plot(A, Ypct, alpha=0.7, label=f"{E_eV/1e6:.3f} MeV")

plt.xlabel("Fragment mass number A")
plt.ylabel("Yield (%)")
plt.title("U-235 mass-yield distribution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()