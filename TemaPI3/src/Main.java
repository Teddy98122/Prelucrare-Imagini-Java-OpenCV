import org.opencv.core.Core;


public class Main {
	 public static void main(String args[]) { 
	      //Loading the OpenCV core library  
	      System.loadLibrary( Core.NATIVE_LIBRARY_NAME ); 
	      ImageManipulation img=new ImageManipulation();
	      img.contour();
	      img.LoadImage();
	      img.SavaImage();
	      img.GrayScaleImg();
	      img.NegativeImage();
	      img.GrayScaleToBinary();
	      img.histogram_equalization2();
	      img.RGBToHSV();
	      img.histogram();
	      img.edgeDetection();
	      img.faceDetection();
	      img.faceDetectionBlackandWhite();
	      img.Sobel();
	      img.noise_reduction();
	      img.image_pyramids();
	      img.distance_transformation();
	      img.laptacian_transformation();
	   } 

}
