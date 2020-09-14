import org.opencv.core.Point;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;




public class ImageManipulation {
	
	//Instantiating the Imagecodecs class 
    Imgcodecs imageCodecs = new Imgcodecs(); 
   
    //Reading the Image from the file  
    String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/image.jpg"; 
    
    public void LoadImage()
    {
    	Mat matrix = imageCodecs.imread(file); 
	    System.out.println("Image Loaded");     
    }
    
    public void noise_reduction()
    {
    	try{ 
           
            System.loadLibrary( Core.NATIVE_LIBRARY_NAME ); 
     
            // Input image 
            Mat source = 
            Imgcodecs.imread("C:/Users/Teddy9812/Desktop/ImaginiPI/blur.jpg", Imgcodecs.IMREAD_ANYCOLOR); 
            Mat destination = new Mat(source.rows(), source.cols(), source.type()); 
     
            // filtering 
            Imgproc.GaussianBlur(source, destination, new Size(0, 0), 10); 
            Core.addWeighted(source, 1.5, destination, -0.5, 0, destination); 
     
            // writing output image 
            Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/blur_noise_rd.jpg", destination); 
         }catch (Exception e) { 
         } 
    }
    
    public void SavaImage()
    {
    	System.out.println("Image Loaded ..........");
        String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image_resaved.jpg"; 
    	Mat matrix = imageCodecs.imread(file); 
    	imageCodecs.imwrite(file2, matrix); 
        System.out.println("Image Saved ............"); 
    }
    
    public void GrayScaleImg()
    {
    	
    	System.out.println("Image Loaded ..........");
        String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image__gray_scale.jpg"; 
    	Mat matrix = imageCodecs.imread(file); 
    	Mat matrix2 = imageCodecs.imread(file2);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2GRAY);
    	imageCodecs.imwrite(file2, matrix2); 
        System.out.println("Image Saved ............"); 
    }
    
    public void NegativeImage()
    {
    	System.out.println("Image Loaded ..........");
        String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image__negative.jpg"; 
    	Mat matrix = imageCodecs.imread(file); 
    	Mat matrix2 = imageCodecs.imread(file2);
    	Mat invertcolormatrix= new Mat(matrix.rows(),matrix.cols(), matrix.type(), new Scalar(255,255,255));
    	Core.subtract(invertcolormatrix, matrix, matrix2);
    	imageCodecs.imwrite(file2, matrix2); 
        System.out.println("Image Saved ............"); 
    }
    public void GrayScaleToBinary()
    {
    	System.out.println("Image Loaded ..........");
        String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image__gray_scale.jpg"; 
        String file3 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image__gray_to_binary.jpg";
    	Mat matrix = imageCodecs.imread(file2); 
    	Mat matrix2 = imageCodecs.imread(file3);
    	Imgproc.threshold(matrix, matrix2, 200, 500, Imgproc.THRESH_BINARY);
    	imageCodecs.imwrite(file3, matrix2); 
        System.out.println("Image Saved ............"); 
    }
    
    public void RGBToHSV()
    {
    	System.out.println("Image Loaded ..........");
        String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/image__HSV.jpg"; 
    	Mat matrix = imageCodecs.imread(file); 
    	Mat matrix2 = imageCodecs.imread(file2);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2HSV);
    	imageCodecs.imwrite(file2, matrix2); 
        System.out.println("Image Saved ............"); 
    }
    
    public void laptacian_transformation()
    {

        //Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/image.jpg";
        Mat src = Imgcodecs.imread(file);

        // Creating an empty matrix to store the result
        Mat dst = new Mat();

        // Applying GaussianBlur on the Image
        Imgproc.Laplacian(src, dst, 10);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/image_lapt_trans.jpg", dst);

        System.out.println("Image Processed");
    }
    
    public void distance_transformation()
    {

        // Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/steve.jpg";
        Mat src = Imgcodecs.imread(file,0);

        // Creating an empty matrix to store the results
        Mat dst = new Mat();
        Mat binary = new Mat();

        // Converting the grayscale image to binary image
        Imgproc.threshold(new Mat(), binary, 100, 255, Imgproc.THRESH_BINARY);

        // Applying distance transform
        Imgproc.distanceTransform(src, dst, Imgproc.DIST_C, 3);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/steve_distance_transform.jpg", dst);

        System.out.println("Image Processed");
    }
    
    public void image_pyramids()
    {
    	// Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/waves.jpg";
        Mat src = Imgcodecs.imread(file);

        // Creating an empty matrix to store the result
        Mat dst = new Mat();

        // Applying pyrUp on the Image
        Imgproc.pyrUp(src, dst, new Size(src.cols()*2,  src.rows()*2), Core.BORDER_DEFAULT);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/waves_pyram.jpg", dst);

        System.out.println("Image Processed");
    }
    
    public void contour()
    {
    	
    	System.out.println("Image Loaded ..........");
    	String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape3.jpg"; 
        String file3 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_grayscale.jpg"; 
    	Mat matrix = imageCodecs.imread(file2); 
    	Mat matrix2 = imageCodecs.imread(file3);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2GRAY);
    	imageCodecs.imwrite(file3, matrix2); 
        System.out.println("Image saved  ............"); 
    	// Loading the OpenCV core library

        // Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_grayscale.jpg";
        Mat src = Imgcodecs.imread(file);

        // Creating an empty matrix to store the result
        Mat dst = new Mat();
        Imgproc.threshold(src, dst, 50, 255, Imgproc.THRESH_BINARY);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_contours.jpg", dst);

        System.out.println("Image Processed"); 
    }
    
    public void Sobel()
    {
    	 // Loading the OpenCV core library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        System.out.println("Image Loaded ..........");
    	String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape3.jpg"; 
        String file3 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_grayscale.jpg"; 
    	Mat matrix = imageCodecs.imread(file2); 
    	Mat matrix2 = imageCodecs.imread(file3);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2GRAY);
    	imageCodecs.imwrite(file3, matrix2); 
        System.out.println("Image saved  ............"); 
        
        // Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_grayscale.jpg";
        Mat src = Imgcodecs.imread(file);

        // Creating an empty matrix to store the result
        Mat dst = new Mat();

        // Applying sobel on the Image
        Imgproc.Sobel(src, dst, -1, 1, 1);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/shape3_sobel.jpg", dst);

        System.out.println("Image processed");
    }
    
    public void edgeDetection()
    {

       
    	System.out.println("Image Loaded ..........");
    	String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape2.jpg"; 
        String file3 = "C:/Users/Teddy9812/Desktop/ImaginiPI/shape2_grayscale.jpg"; 
    	Mat matrix = imageCodecs.imread(file2); 
    	Mat matrix2 = imageCodecs.imread(file3);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2GRAY);
    	imageCodecs.imwrite(file3, matrix2); 
        System.out.println("Image saved  ............"); 
    	
        Mat edges = new Mat();

        // Detecting the edges
        Imgproc.Canny(matrix2, edges, 60, 60*3);

        // Writing the image
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/shape2.edge.jpg", edges);
        System.out.println("Image Loaded");
    }
    
    public void faceDetection()
    {
    	
    	// Citirea imaginii
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/face_group.jpg";
        Mat src = Imgcodecs.imread(file);

        // Instantierea CascadeClassifier
        String xmlFile = "C:/Users/Teddy9812/eclipse-workspace/Tema1PI/src/lbpcascade_frontalface.xml";
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        // Detectarea fetelor
        MatOfRect faceDetections = new MatOfRect();
        classifier.detectMultiScale(src, faceDetections);
        System.out.println(String.format("Detected %s faces", 
           faceDetections.toArray().length));
  
        for (Rect rect : faceDetections.toArray()) {
        	  Imgproc.rectangle(
              src,                                               
              new Point(rect.x, rect.y),                            
              new Point(rect.x + rect.width, rect.y + rect.height), 
              new Scalar(0, 0, 255),
              3                                                    
           ); 
        }
        

        // Scrierea imaginii
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/face_group_detected.jpg", src);

        System.out.println("Image Processed");
    }
   
    public void faceDetectionBlackandWhite()
    {
    	
    	System.out.println("Image Loaded ..........");
    	String file2 = "C:/Users/Teddy9812/Desktop/ImaginiPI/face_group.jpg"; 
        String file3 = "C:/Users/Teddy9812/Desktop/ImaginiPI/face_group_grayscale.jpg"; 
    	Mat matrix = imageCodecs.imread(file2); 
    	Mat matrix2 = imageCodecs.imread(file3);
    	Imgproc.cvtColor(matrix, matrix2, Imgproc.COLOR_RGB2GRAY);
    	imageCodecs.imwrite(file3, matrix2); 
        System.out.println("Image saved  ............"); 
    	
    	// Citirea imaginii
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/face_group_grayscale.jpg";
        Mat src = Imgcodecs.imread(file);

        // Instantierea CascadeClassifier
        String xmlFile = "C:/Users/Teddy9812/eclipse-workspace/Tema1PI/src/lbpcascade_frontalface.xml";
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        // Detectarea fetelor
        MatOfRect faceDetections = new MatOfRect();
        classifier.detectMultiScale(src, faceDetections);
        System.out.println(String.format("Detected %s faces", 
           faceDetections.toArray().length));
  
        for (Rect rect : faceDetections.toArray()) {
      	  Imgproc.rectangle(
            src,                                               
            new Point(rect.x, rect.y),                            
            new Point(rect.x + rect.width, rect.y + rect.height), 
            new Scalar(0, 0, 255),
            3                                                    
         ); 
      }
        
     // Scrierea imaginii
        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/face_group_grayscale_detected.jpg", src);

        System.out.println("Image Processed");
    }
 
    public void histogram_equalization2()
    {
    	// Reading the Image from the file and storing it in to a Matrix object
        String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/earth.jpg";

        // Load the image
        Mat img = Imgcodecs.imread(file);

        // Creating an empty matrix
        Mat equ = new Mat();
        img.copyTo(equ);

        // Applying blur
        Imgproc.blur(equ, equ, new Size(3, 3));

        // Applying color
        Imgproc.cvtColor(equ, equ, Imgproc.COLOR_BGR2YCrCb);
        List<Mat> channels = new ArrayList<Mat>();

        // Splitting the channels
        Core.split(equ, channels);

        // Equalizing the histogram of the image
        Imgproc.equalizeHist(channels.get(0), channels.get(0));
        Core.merge(channels, equ);
        Imgproc.cvtColor(equ, equ, Imgproc.COLOR_YCrCb2BGR);

        Mat gray = new Mat();
        Imgproc.cvtColor(equ, gray, Imgproc.COLOR_BGR2GRAY);
        Mat grayOrig = new Mat();
        Imgproc.cvtColor(img, grayOrig, Imgproc.COLOR_BGR2GRAY);

        Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/earth_eq_hist.jpg", equ);
        System.out.println("Image Processed");
     }
    
    public void histogram() {
		// Reading the Image from the file and storing it in to a Matrix object
	      String file ="C:/Users/Teddy9812/Desktop/ImaginiPI/image.jpg";

	      // Load the image
	      Mat img = Imgcodecs.imread(file);

	      // Creating an empty matrix
	      Mat equ = new Mat();
	      img.copyTo(equ);

	      // Applying blur
	      Imgproc.blur(equ, equ, new Size(3, 3));

	      // Applying color
	      Imgproc.cvtColor(equ, equ, Imgproc.COLOR_BGR2YCrCb);
	      ArrayList<Mat> channels = new ArrayList<Mat>();

	      // Splitting the channels
	      Core.split(equ, channels);

	      // Equalizing the histogram of the image
	      Imgproc.equalizeHist(channels.get(0), channels.get(0));
	      Core.merge(channels, equ);
	      Imgproc.cvtColor(equ, equ, Imgproc.COLOR_YCrCb2BGR);

	      Mat gray = new Mat();
	      Imgproc.cvtColor(equ, gray, Imgproc.COLOR_BGR2GRAY);
	      Mat grayOrig = new Mat();
	      Imgproc.cvtColor(img, grayOrig, Imgproc.COLOR_BGR2GRAY);

	      Imgcodecs.imwrite("C:/Users/Teddy9812/Desktop/ImaginiPI/image_histogram.jpg", equ);
	      System.out.println("Image Procesata");
	   }
}
