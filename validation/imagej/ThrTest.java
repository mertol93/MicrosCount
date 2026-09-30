import ij.process.AutoThresholder;
import java.io.*;
import java.util.*;
public class ThrTest {
  public static void main(String[] a) throws Exception {
    BufferedReader r = new BufferedReader(new FileReader(a[0]));
    String line; AutoThresholder t = new AutoThresholder();
    StringBuilder sb = new StringBuilder();
    while ((line = r.readLine()) != null) {
      String[] p = line.trim().split(",");
      int[] h = new int[p.length];
      for (int i = 0; i < p.length; i++) h[i] = Integer.parseInt(p[i]);
      int d = t.getThreshold(AutoThresholder.Method.Default, h.clone());
      int iso = t.getThreshold(AutoThresholder.Method.IJ_IsoData, h.clone());
      sb.append(d).append(",").append(iso).append("\n");
    }
    System.out.print(sb);
  }
}
