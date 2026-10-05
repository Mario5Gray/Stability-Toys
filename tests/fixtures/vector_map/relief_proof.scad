// S1.3 relief proof (STABL-nygbbrrn). Hand-written. Not the S3.4 SCAD generator.
// Extrude one traced silhouette layer. Spec 6.5: import with center=false.
// Set both values on the command line:
//   openscad -o out.stl -D 'svg="layer.svg"' -D 'thickness=2' relief_proof.scad

svg = "";        // Path to the layer SVG.
thickness = 2;   // Silhouette thickness S in mm.

linear_extrude(height = thickness) import(svg, center = false);
