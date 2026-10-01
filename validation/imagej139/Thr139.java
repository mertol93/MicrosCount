import ij.process.*; import java.io.*;
// ImageJ 1.39u levels for histograms: "mask" = Thresholder (getAutoThreshold of the raw histogram),
// "auto" = ThresholdAdjuster (getAutoThreshold after the setHistogram mode clipping).
public class Thr139 {
  public static void main(String[] a) throws Exception {
    BufferedReader r = new BufferedReader(new FileReader(a[0])); String line; ByteProcessor bp = new ByteProcessor(1,1);
    StringBuilder sb = new StringBuilder();
    while ((line = r.readLine()) != null) {
      String[] p = line.trim().split(","); int[] h = new int[p.length];
      for (int i=0;i<p.length;i++) h[i]=Integer.parseInt(p[i]);
      int mask = bp.getAutoThreshold(h.clone());
      int[] c = h.clone(); int mode=0, maxCount=0;
      for (int i=0;i<c.length;i++) if (c[i]>maxCount) {maxCount=c[i]; mode=i;}
      int max2=0; for (int i=0;i<c.length;i++) if (c[i]>max2 && i!=mode) max2=c[i];
      if ((maxCount>(max2*2)) && (max2!=0)) c[mode]=(int)(max2*1.5);
      int auto = bp.getAutoThreshold(c);
      sb.append(mask).append(",").append(auto).append("\n");
    }
    System.out.print(sb); System.exit(0);
  }
}
