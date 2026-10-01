import ij.*; import ij.process.*; import ij.plugin.filter.RankFilters; import ij.measure.Measurements; import ij.io.FileSaver;
// Noursadeghi et al. (2008) run with ImageJ 1.39u's own classes.
// route "auto": Image>Adjust>Threshold>Auto (ThresholdAdjuster: mode-clipped histogram) then Apply/Convert to Mask
// route "mask": Process>Binary>Convert to Mask with no threshold set (Thresholder.autoThreshold: raw histogram)
// Both keep pixels >= level. Masks are applied to the original rel A image; zero bin dropped (histogram export).
public class Paper139 {
  static int level(ImageProcessor ip, String route) {
    ImageStatistics stats = ImageStatistics.getStatistics(ip, Measurements.AREA+Measurements.MIN_MAX+Measurements.MODE, null);
    if (route.equals("mask")) return ((ByteProcessor)ip).getAutoThreshold();
    int[] h = stats.histogram; int max2 = 0;                       // ThresholdAdjuster.setHistogram (1.39u)
    for (int i=0;i<stats.nBins;i++) if (h[i]>max2 && i!=stats.mode) max2=h[i];
    int hmax = stats.maxCount;
    if ((hmax>(max2*2)) && (max2!=0)) { hmax=(int)(max2*1.5); h[stats.mode]=hmax; }
    return ip.getAutoThreshold(h);                                  // ThresholdAdjuster.autoSetLevels
  }
  public static void main(String[] a) {
    String route = a[3];
    ImagePlus N = IJ.openImage(a[0]), T = IJ.openImage(a[1]);
    ImageProcessor nf = N.getProcessor().duplicate(), tf = T.getProcessor().duplicate();
    new RankFilters().rank(nf, 1, RankFilters.MEDIAN); new RankFilters().rank(tf, 1, RankFilters.MEDIAN);
    new FileSaver(new ImagePlus("Nf", nf)).saveAsTiff(a[2]+"_Nf139.tif");
    new FileSaver(new ImagePlus("Tf", tf)).saveAsTiff(a[2]+"_Tf139.tif");
    int ln = level(nf, route), lt = level(tf, route);
    ImageProcessor traw = T.getProcessor();
    int w = nf.getWidth(), hh = nf.getHeight();
    ByteProcessor nm = new ByteProcessor(w,hh), cm = new ByteProcessor(w,hh);
    double ns=0, cs=0; long nc=0, cc=0;
    for (int y=0;y<hh;y++) for (int x=0;x<w;x++) {
      int mN = nf.get(x,y)>=ln ? 255 : 0, mT = tf.get(x,y)>=lt ? 255 : 0;      // Convert to Mask lut
      int mC = Math.max(0, mT - mN);                                           // Image Calculator: Subtract
      int v = traw.get(x,y);
      int vN = mN & v, vC = mC & v;                                            // Image Calculator: AND with original
      if (mN>0) nm.set(x,y,255); if (mC>0) cm.set(x,y,255);
      if (vN>0) { ns+=vN; nc++; }                                              // histogram without the zero bin
      if (vC>0) { cs+=vC; cc++; }
    }
    new FileSaver(new ImagePlus("Nm", nm)).saveAsTiff(a[2]+"_Nmask139.tif");
    new FileSaver(new ImagePlus("Cm", cm)).saveAsTiff(a[2]+"_Cmask139.tif");
    System.out.println("RESULT139,"+route+","+ln+","+lt+","+nc+","+(ns/nc)+","+cc+","+(cs/cc)+","+((ns/nc)/(cs/cc)));
    System.exit(0);
  }
}
