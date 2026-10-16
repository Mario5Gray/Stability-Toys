// st-vector-map S3.4 relief source (STABL-gorlibrt). Generated file.
// Edit the parameters above the modules, then reopen the file in OpenSCAD.
// Layer imports are relative to this file, uncentered, in millimetres.
// Top heights: backing B, silhouette B + S, structure B + S + H, detail B + S + D.

B = 0; // backing thickness. 0 gives no backing.
S = 1.2; // silhouette thickness.
H = 0.6; // structure height above the silhouette.
D = 0.2; // engraving depth below the silhouette top.
OVERLAP = 0.01; // bounded support overlap. Z never goes below 0.
RECESS_BRIDGE = 0.002; // half side of the uncut square where the recess meets the outline at a point.
RECESS_BRIDGES = [[39.84375, 39.453125], [54.296875, 72.65625], [64.0625, 46.484375], [77.34375, 0.390625], [80.078125, 0], [92.1875, 95.3125], [97.265625, 100]]; // those points, in millimetres.

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
      import("out.layers/detail.svg", center=false);
      silhouette_2d();
    }
    structure_2d();
  }
}

module engrave_2d() {
  difference() {
    import("out.layers/detail.svg", center=false);
    import("out.layers/structure.svg", center=false);
    for (p = RECESS_BRIDGES) translate(p) square(2 * RECESS_BRIDGE, center = true);
  }
}

module relief() {
  if (B > 0) {
    linear_extrude(height = B) square([100, 100]);
  }
  difference() {
    translate([0, 0, max(0, B - OVERLAP)])
      linear_extrude(height = S + min(B, OVERLAP)) silhouette_2d();
    translate([0, 0, B + S - D])
      linear_extrude(height = D + OVERLAP) engrave_2d();
  }
  translate([0, 0, max(0, B + S - OVERLAP)])
    linear_extrude(height = H + min(B + S, OVERLAP)) structure_2d();
}

relief();
