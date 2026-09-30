import ij.*; import ij.process.*; import ij.plugin.filter.RankFilters; import ij.io.FileSaver;
public class PaperPipeline {
  static ImageProcessor median(ImageProcessor ip){ ImageProcessor d=ip.duplicate(); new RankFilters().rank(d,1,RankFilters.MEDIAN); return d; }
  public static void main(String[] a){
    ImagePlus N=IJ.openImage(a[0]), T=IJ.openImage(a[1]);
    ImageProcessor nf=median(N.getProcessor()), tf=median(T.getProcessor());
    new FileSaver(new ImagePlus("Nf",nf)).saveAsTiff(a[2]+"_Nf.tif");
    new FileSaver(new ImagePlus("Tf",tf)).saveAsTiff(a[2]+"_Tf.tif");
    nf.setAutoThreshold("Default dark"); tf.setAutoThreshold("Default dark");
    double nl=nf.getMinThreshold(), nu=nf.getMaxThreshold(), tl=tf.getMinThreshold(), tu=tf.getMaxThreshold();
    int w=nf.getWidth(), h=nf.getHeight(); ByteProcessor nm=new ByteProcessor(w,h), cm=new ByteProcessor(w,h);
    double ns=0, cs=0; long nc=0, cc=0; ImageProcessor t=T.getProcessor();
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
      int nv=nf.get(x,y), tv=tf.get(x,y);
      boolean inN = nv>=nl && nv<=nu;               // Convert to Mask: pixels within the threshold range
      boolean inT = tv>=tl && tv<=tu;
      int sub = Math.max(0,(inT?255:0)-(inN?255:0)); // Image Calculator, Subtract (8-bit clamps at 0)
      if(inN){ nm.set(x,y,255); ns+=t.get(x,y); nc++; }
      if(sub>0){ cm.set(x,y,255); cs+=t.get(x,y); cc++; }
    }
    new FileSaver(new ImagePlus("Nmask",nm)).saveAsTiff(a[2]+"_Nmask.tif");
    new FileSaver(new ImagePlus("Cmask",cm)).saveAsTiff(a[2]+"_Cmask.tif");
    System.out.println("RESULT,"+nl+","+nu+","+tl+","+tu+","+nc+","+(ns/nc)+","+cc+","+(cs/cc)+","+((ns/nc)/(cs/cc))); System.exit(0);
  }
}
