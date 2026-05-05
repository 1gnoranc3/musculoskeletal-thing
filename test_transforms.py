import opensim as osim, pandas as pd, tempfile, sys
from pathlib import Path
sys.path.insert(0, '/Users/ignorance/PycharmProjects/diplom4ik')
from run_pipeline import rename_trc_markers

MODEL_PATH   = '/Users/ignorance/PycharmProjects/diplom4ik/Rajagopal_DM_scaled.osim'
IK_TASKS_XML = '/Users/ignorance/PycharmProjects/diplom4ik/ik_tasks.xml'
TRC_ORIG = Path('/Users/ignorance/PycharmProjects/diplom4ik/Synchronized Motion Data/Static Trials/Video Motion Data/DM_static_overground1_new.trc')

# Варианты: None = без трансформации; tuple = (xi, xs, yi, ys, zi, zs)
VARIANTS = {
    "GC(Y,Z,-X)[текущий]":  (1,1, 2,1, 0,-1),
    "GC(Y,Z,+X)":           (1,1, 2,1, 0,+1),
    "GC(-Y,Z,+X)":          (1,-1, 2,1, 0,+1),
    "GC(-Y,Z,-X)":          (1,-1, 2,1, 0,-1),
    "без трансф":            None,
}

def make_trc(src, dst, transform):
    rename_trc_markers(src, dst)  # rename + current transform
    if transform is None or transform == (1,1, 2,1, 0,-1):
        return
    # override with new transform
    with open(dst) as f: lines = f.readlines()
    with open(src) as f: orig = f.readlines()
    xi,xs, yi,ys, zi,zs = transform
    new_lines = lines[:5]
    for line in orig[5:]:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 3: new_lines.append(line); continue
        out = cols[:2]; i = 2
        while i + 2 < len(cols):
            try:
                v = [float(cols[i]), float(cols[i+1]), float(cols[i+2])]
                out += [f'{v[xi]*xs:.6f}', f'{v[yi]*ys:.6f}', f'{v[zi]*zs:.6f}']
            except: out += cols[i:i+3]
            i += 3
        out += cols[i:]; new_lines.append('\t'.join(out)+'\n')
    # Apply marker renaming to the output
    import re
    hdr_line = new_lines[3]
    from run_pipeline import MARKER_MAP
    parts = hdr_line.rstrip('\n').split('\t')
    for j,p in enumerate(parts):
        if p.strip() in MARKER_MAP: parts[j] = MARKER_MAP[p.strip()]
    new_lines[3] = '\t'.join(parts)+'\n'
    with open(dst,'w') as f: f.writelines(new_lines)

def run_ik(trc_path, out_mot):
    model = osim.Model(MODEL_PATH); model.initSystem()
    with open(trc_path) as f: lines = f.readlines()
    t_s = t_e = None
    for line in lines:
        cols = line.strip().split('\t')
        if len(cols)>1:
            try: t=float(cols[1]); t_s=t_s or t; t_e=t
            except: pass
    ik = osim.InverseKinematicsTool()
    ik.setModel(model); ik.setMarkerDataFileName(str(trc_path))
    ik.setStartTime(t_s); ik.setEndTime(t_e)
    ik.setOutputMotionFileName(out_mot); ik.set_report_errors(False)
    ik.set_IKTaskSet(osim.IKTaskSet(IK_TASKS_XML)); ik.run()

print(f"{'Вариант':<25} {'tilt':>7} {'list':>7} {'rot':>7} {'hip_f':>7} {'knee':>7}")
print("-"*65)
for name, transform in VARIANTS.items():
    tmp = Path(tempfile.mktemp(suffix='.trc'))
    out = f'/tmp/ikt_{name[:6]}.mot'
    try:
        make_trc(TRC_ORIG, tmp, transform)
        run_ik(tmp, out)
        with open(out) as f: ll = f.readlines()
        he = next(i for i,l in enumerate(ll) if l.strip().lower()=='endheader')+1
        df = pd.read_csv(out, sep='\t', skiprows=he)
        print(f"{name:<25} {df.pelvis_tilt.mean():>7.1f} {df.pelvis_list.mean():>7.1f} {df.pelvis_rotation.mean():>7.1f} {df.hip_flexion_r.mean():>7.1f} {df.knee_angle_r.mean():>7.1f}")
    except Exception as e: print(f"{name:<25} ERROR: {e}")
    finally: tmp.unlink(missing_ok=True)
