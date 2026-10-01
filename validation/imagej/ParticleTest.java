import ij.*; import ij.process.*; import ij.plugin.filter.ParticleAnalyzer; import ij.measure.*;
// Analyze > Analyze Particles (size 0-Infinity, circularity 0-1) on a 0/255 mask, ImageJ 1.54
public class ParticleTest {
  public static void main(String[] a) {
    ImagePlus imp = IJ.openImage(a[0]);
    ImageProcessor ip = imp.getProcessor();
    ip.setThreshold(255, 255, ImageProcessor.NO_LUT_UPDATE);
    ResultsTable rt = new ResultsTable();
    int meas = Measurements.AREA | Measurements.PERIMETER | Measurements.SHAPE_DESCRIPTORS;
    ParticleAnalyzer pa = new ParticleAnalyzer(ParticleAnalyzer.RECORD_STARTS, meas, rt, 0, Double.POSITIVE_INFINITY, 0.0, 1.0);
    pa.setHideOutputImage(true);
    pa.analyze(imp, ip);
    StringBuilder sb = new StringBuilder();
    for (int i=0;i<rt.size();i++) sb.append("P,"+(int)rt.getValue("XStart",i)+","+(int)rt.getValue("YStart",i)+","+rt.getValue("Area",i)+","+rt.getValue("Perim.",i)+","+rt.getValue("Circ.",i)+"\n");
    System.out.print(sb);
    System.exit(0);
  }
}
