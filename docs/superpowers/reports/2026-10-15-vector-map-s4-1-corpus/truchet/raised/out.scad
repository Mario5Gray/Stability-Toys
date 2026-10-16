// st-vector-map S3.4 relief source (STABL-gorlibrt). Generated file.
// Edit the parameters above the modules, then reopen the file in OpenSCAD.
// Layer imports are relative to this file, uncentered, in millimetres.
// Top heights: backing B, silhouette B + S, structure B + S + H, detail B + S + D.

B = 0; // backing thickness. 0 gives no backing.
S = 1.2; // silhouette thickness.
H = 0.6; // structure height above the silhouette.
D = 0.2; // detail height above the silhouette.
OVERLAP = 0.01; // bounded support overlap. Z never goes below 0.
DETAIL_GROW = 0.002; // detail grows by this much so it never meets structure at a point.

module silhouette_2d() {
  import("out.layers/silhouette.svg", center=false);
}

module structure_2d() {
  intersection() {
    import("out.layers/structure.svg", center=false);
    silhouette_2d();
  }
}

module detail_2d() {
  difference() {
    intersection() {
      offset(delta = DETAIL_GROW) import("out.layers/detail.svg", center=false);
      silhouette_2d();
    }
    offset(delta = -DETAIL_GROW) structure_2d();
  }
}

module relief() {
  if (B > 0) {
    linear_extrude(height = B) square([100, 100]);
  }
  translate([0, 0, max(0, B - OVERLAP)])
    linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();
  translate([0, 0, max(0, B + S - OVERLAP)])
    linear_extrude(height = H + min(B + S, OVERLAP)) structure_2d();
  translate([0, 0, max(0, B + S - OVERLAP)])
    linear_extrude(height = D + min(B + S, OVERLAP)) detail_2d();
}

relief();
