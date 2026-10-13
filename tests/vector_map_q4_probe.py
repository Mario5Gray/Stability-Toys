from __future__ import annotations
import json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import vector_map
from tests.test_vector_map_openscad import mesh_components
from tests.test_vector_map_layer_export import geometry_masks

OPENSCAD = "/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD"

def save_mask(path, mask):
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)

def metrics(stl):
    txt = stl.read_text()
    triples = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
    if not triples:
        return {"parts": 0, "bounds": None, "components": []}
    corners = np.array([[float(v) for v in row] for row in triples])
    vertices, index = np.unique(corners.round(6), axis=0, return_inverse=True)
    parts = mesh_components(vertices, index.reshape(-1, 3))
    return {"parts": len(parts), "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
            "components": sorted([[a.tolist(), b.tolist(), int(g)] for a,b,g in parts],
                                 key=lambda x: (x[0], x[1], x[2]))}

def render(directory, stem, expression):
    scad = directory / (stem + ".scad")
    stl = directory / (stem + ".stl")
    scad.write_text("linear_extrude(height=2) " + expression + "\n")
    command = [OPENSCAD, "-o", str(stl), str(scad)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(json.dumps({"cmd": command, "stderr": result.stderr}))
    return {"cmd": command, "mesh": metrics(stl)}

def imp(name):
    return 'import("out.layers/' + name + '.svg", center=false);'

def bool_structure():
    return "intersection() {" + imp("structure") + imp("silhouette") + "}"

def bool_detail():
    return "difference() {intersection() {" + imp("detail") + imp("silhouette") + "} " + bool_structure() + "}"

def check_case(root, name, masks, scale):
    directory = root / name
    directory.mkdir()
    h,w = masks["silhouette"].shape
    source = directory / "source.png"
    Image.fromarray(np.full((h,w,3), 90, dtype=np.uint8)).save(source)
    for role, mask in masks.items(): save_mask(directory / (role + ".png"), mask)
    args = [str(source), str(directory / "out.svg"), "--input-kind", "image",
            "--mask", str(directory / "silhouette.png"), "--width-mm", str(w*scale),
            "--layers", "silhouette,structure,detail",
            "--structure-map", str(directory / "structure.png"),
            "--detail-map", str(directory / "detail.png"), "--json"]
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err): code = vector_map.main(args)
    result = json.loads(out.getvalue())
    if code: raise RuntimeError(json.dumps({"args":args,"code":code,"result":result,"stderr":err.getvalue()}))
    paths = result["artifacts"]["layers"]
    if set(paths) != {"silhouette","structure","detail"}:
        raise RuntimeError(json.dumps({"case":name,"paths":paths}))
    data = {"name":name,"scale":scale,"published":paths,"renders":{}}
    expressions = {"silhouette":imp("silhouette"),"structure_direct":imp("structure"),
                   "structure_bool":bool_structure(),"detail_direct":imp("detail"),
                   "detail_bool":bool_detail()}
    for label, expression in expressions.items():
        data["renders"][label] = render(directory,label,expression)
    return data

def near_case():
    h,w=32,44
    s=np.zeros((h,w),bool); s[2:30,2:42]=True
    structure=np.zeros_like(s); structure[5:27,3:21]=True
    detail=np.zeros_like(s); detail[5:27,21:40]=True
    return {"silhouette":s,"structure":structure,"detail":detail}

def stair_case():
    h,w=34,48
    s=np.zeros((h,w),bool); s[2:32,2:46]=True
    structure=np.zeros_like(s); detail=np.zeros_like(s)
    for y in range(5,29):
        edge=16+(y-5)//4
        structure[y,4:edge]=True
        detail[y,edge:43]=True
    return {"silhouette":s,"structure":structure,"detail":detail}

def crossing_case():
    h,w=32,44
    s=np.zeros((h,w),bool); s[:, :41]=True; s[8:24,12:32]=False; s[13:19,19:25]=True
    structure=np.zeros_like(s); structure[15:18,:43]=True
    detail=np.zeros_like(s); detail[:,21:24]=True
    return {"silhouette":s,"structure":structure,"detail":detail}

def border_case():
    h,w=30,40
    s=np.zeros((h,w),bool); s[0:28,0:35]=True; s[8:19,10:23]=False
    structure=np.zeros_like(s); structure[2:5,0:40]=True; structure[20:30,2:6]=True
    detail=np.zeros_like(s); detail[5:25,22:26]=True
    return {"silhouette":s,"structure":structure,"detail":detail}

def open_closed(root):
    out={}
    for label, path in {"open":"M1,1 L12,1 L12,12","closed":"M1,1 L12,1 L12,12 Z"}.items():
        directory=root/("path_"+label);directory.mkdir()
        (directory/"shape.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16mm" height="16mm" viewBox="0 0 16 16"><path d="'+path+'" fill="black"/></svg>')
        out[label]=render(directory,label,'import("shape.svg", center=false);')
    return out

def main():
    version=subprocess.run([OPENSCAD,"--version"],capture_output=True,text=True).stderr.strip()
    with tempfile.TemporaryDirectory(prefix="q4-full-",dir="/private/tmp") as tmp:
        root=Path(tmp)
        cases=[("compound_nested",geometry_masks(),0.37),("crossing",crossing_case(),0.37),
               ("border",border_case(),25.4/96),("near",near_case(),0.37),("stair",stair_case(),0.37)]
        rows=[check_case(root,*case) for case in cases]
        path_probe=open_closed(root)
        for row in rows:
            renders=row["renders"]
            for a,b in (("structure_direct","structure_bool"),("detail_direct","detail_bool")):
                left,right=renders[a]["mesh"],renders[b]["mesh"]
                row[a+"_equals_"+b]=left==right
        for row in rows:
            assert row["structure_direct_equals_structure_bool"], row["name"] + " structure"
            assert row["detail_direct_equals_detail_bool"], row["name"] + " detail"
        assert path_probe["open"]["mesh"]["bounds"] != path_probe["closed"]["mesh"]["bounds"]
        print(json.dumps({"version":version,"cases":rows,"open_closed":path_probe},indent=2))
if __name__=="__main__": main()
