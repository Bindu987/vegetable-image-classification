# vegetable-image-classification

# **🥦 Vegetable Image Classification and Detection using Deep Learning**

This project presents a comparative study of deep learning-based models for classifying and detecting vegetables in images. Implemented in **MATLAB**, the models are trained on a curated dataset of 9 vegetable categories using both **custom CNN architectures** and **pretrained transfer learning models**. Real-time object detection is implemented using **YOLOv4**.

Developed as a course project for **ECE 172 at California State University, Fresno**, this work demonstrates practical applications of deep learning in **agricultural automation**.

---

## 🧠 Models Implemented

| Model           | Type               | Accuracy     |
|----------------|--------------------|--------------|
| Custom DCNN     | Custom CNN         | 96.89%       |
| SqueezeNet      | Transfer Learning  | 99.94%       |
| ResNet-50       | Transfer Learning  | 99.94%       |
| Inception-v3    | Transfer Learning  | **100%**     |
| YOLOv4          | Object Detection   | **AP = 1.0** |

---

## 📦 Dataset

- **Source**: [Kaggle: Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Categories (9)**: Bottle Gourd, Brinjal, Broccoli, Carrot, Cauliflower, Cucumber, Potato, Pumpkin, Radish
- **Total images**: ~9000
- **Annotation**: Manual bounding boxes via MATLAB Image Labeler for YOLOv4

---

## 🛠️ Tools and Technologies

- **Language**: MATLAB R2023a
- **Toolboxes Used**:
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - Computer Vision Toolbox
- **Add-ons**:
  - Pretrained Models: ResNet-50, Inception-v3, SqueezeNet
  - YOLOv4 Support Package

---

## 🧪 Methodology

1. **Data Preprocessing**:
   - Image resizing to match model input
   - Augmentation for detection tasks (brightness, mirroring, etc.)

2. **Model Training**:
   - Custom DCNN: 3 Conv layers → BatchNorm → ReLU → MaxPool → Fully Connected → Softmax
   - Pretrained models: Last layers replaced for 9-class classification
   - 80-20 stratified training-validation split
   - Optimizers: SGDM / Adam

3. **Detection Pipeline**:
   - Image annotation with MATLAB Image Labeler
   - YOLOv4 trained on ~900 annotated images
   - Average Precision evaluated using PR curves

---

## 📁 Project Structure

```
vegetable-image-classification/
│
├── docs/
│   ├── ieee_format_report.pdf
│   └── results_and_project_report.pdf
│
├── src/
│   ├── Custom DCNN/
│   ├── SqueezeNet/
│   ├── ResNet-50/
│   ├── Inception-v3/
│   └── YOLOv4/
│
├── dataset/        
│
└── README.md
```

---

## 📸 Results Overview

- Inception-v3 showed perfect classification performance with 100% accuracy.
- SqueezeNet and ResNet-50 performed comparably with 99.94%.
- YOLOv4 achieved **Average Precision (AP) = 1.0** on test set.
- All pretrained models demonstrated better convergence and generalization than the custom DCNN.
- In **results_and_project_report doc**, could see results and confusion matrices and data analysis.

---

## 🚀 Future Enhancements

- Expand dataset with more classes and samples
- Deploy models on edge devices (Jetson Nano, Raspberry Pi)
- Upgrade detection framework to **YOLOv8**
- Explore ensemble learning for better reliability

---

## 👥 Authors

- **Bindu Sree Chandu**  
  Department of Electrical and Computer Engineering  
  California State University, Fresno  

- **Govardhana Kondapaturi**  
  Department of Electrical and Computer Engineering  
  California State University, Fresno  

---

## 📄 License

This project is for academic purposes. If you wish to reuse or adapt it, please credit the authors.

---

## ⭐ GitHub Tips

If you like this project or find it useful, consider giving it a ⭐ on GitHub!
