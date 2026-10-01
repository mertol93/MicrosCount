package com.sun.image.codec.jpeg;
public interface JPEGImageEncoder {
  JPEGEncodeParam getDefaultJPEGEncodeParam(java.awt.image.BufferedImage bi);
  void setJPEGEncodeParam(JPEGEncodeParam p);
  void encode(java.awt.image.BufferedImage bi) throws java.io.IOException;
  void encode(java.awt.image.BufferedImage bi, JPEGEncodeParam p) throws java.io.IOException;
}
