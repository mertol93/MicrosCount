import ij.*; import ij.process.*; import ij.plugin.filter.BackgroundSubtracter; import ij.io.FileSaver;
// Process > Subtract Background... (rolling ball, dialog defaults) with ImageJ 1.54's own class
public class RollingBallTest {
  public static void main(String[] a) {
    ImagePlus imp = IJ.openImage(a[0]);
    ImageProcessor ip = imp.getProcessor();
    new BackgroundSubtracter().rollingBallBackground(ip, Double.parseDouble(a[1]), false, false, false, true, true);
    new FileSaver(imp).saveAsTiff(a[2]);
    System.exit(0);
  }
}
