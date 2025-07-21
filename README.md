# **Low-Dose CT Scan Denoising with RED-CNN**  

## **Overview**  
This project focuses on enhancing the quality of low-dose CT scan images affected by Poisson and periodic noise. Using a **Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)**, the system provides denoised CT scans to improve medical diagnosis while reducing the need for high-dose radiation exposure.  

To ensure accessibility, a **web-based application platform** has been developed that allows users to upload and process CT scans seamlessly. The backend is built with **Django**, while the frontend uses **HTML, CSS, and JavaScript**, ensuring a responsive and user-friendly experience.  

## **Features**  
- **Real-time Denoising**: Processes CT scans with RED-CNN to remove Poisson and periodic noise.  
- **Web-based Platform**: Users can upload CT scans in **DICOM or IMA format** for denoising.  
- **Quantitative Evaluation**: The system calculates **PSNR, SNR, and SSIM** to assess denoising performance.  
- **High-Resolution Output**: Enhanced CT scans can be downloaded in high-resolution **IMA format**.  
- **Optimized Performance**: Designed for minimal latency and high processing speed.  

## **System Workflow**  
1. Users upload a noisy CT scan in DICOM or IMA format.  
2. The backend (Django) processes the image using the RED-CNN model.  
3. The model removes noise and generates a denoised CT scan.  
4. The processed image is sent back to the frontend for visualization.  
5. The system calculates and displays quantitative metrics (PSNR, SNR, SSIM).  
6. Users can download the enhanced CT scan in high resolution.  

## **Installation & Setup**  

### **1. Requirements**  
- Python 3.8+  
- Django  
- TensorFlow / PyTorch (for RED-CNN)  
- NumPy, OpenCV, Pydicom (for handling DICOM/IMA files)  
- HTML, CSS, JavaScript (for frontend)  

### **2. Installation Steps**  
```sh
git clone https://github.com/your-repo/ct-denoising.git
cd ct-denoising
pip install -r requirements.txt
```

### **3. Running the Web Application**  
```sh
python manage.py runserver
``` 
## **Demo Video**  
[Watch the Demo](https://drive.google.com/file/d/1-_est69XeAdbp_6C9H2lsAKPiJOw_M5d/view?usp=drive_link)  

## **Contributors**  
- Joel Ebenezer
- Tharun Nivek
- Mukhil Adinath
- Prithiv Varma  

## **License**  
MIT License  
