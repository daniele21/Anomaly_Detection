# Master Thesis
## Anomaly Detection with Ganomaly: patch-wise analysis and transfer learning

### Description
Anomaly detection on images permits to identify an abnormal image. In general, the dataset are very unbalanced, providing very few occurences of abnormal images, for this reason, this project wants to provide a model based on SEMI-SUPERVISED LEARNING: training the model on NORMAL images, it is possible to detect the anomalies as images that are not conform to the normal standard.

### Goal
Providing an anomaly detection model, trained on normal samples, able to detect when an image is abnormal and it can detect the anomalous region inside the image itself. It is a DEEP NEURAL NETWORK model which solves the segmentation problem, providing for each image the normal region and abnormal region detected.
#### The provided models can infere on images of every size, thanks to its PATCH-WISE TRAINING

### Model Details:
In this project two baseline model are considered, in order to compare them with the ones proposed by this study.
  - Baseline 1: Convolutional Autoencoder
  - Baseline 2: CNN fully connected
  
The proposed models are:
  - Patch-Ganomaly -> it introduce the patch-wise training technique onto Ganomaly model

![PG](https://github.com/daniele21/Anomaly_Detection/blob/master/Results/ganomaly.png)

  - TL-Ganomaly    -> it uses the patch-wise training and transfer learning techniques onto Ganomaly
![TL](https://github.com/daniele21/Anomaly_Detection/blob/master/Results/Tl-arch_2.png)  

### Results:
![exp](https://github.com/daniele21/Anomaly_Detection/blob/master/Results/Experiments.png)

<img src="https://github.com/daniele21/Anomaly_Detection/blob/master/Results/exp1.png" width="270">     <img src="https://github.com/daniele21/Anomaly_Detection/blob/master/Results/exp2.png" width="270">     <img src="https://github.com/daniele21/Anomaly_Detection/blob/master/Results/exp3.png" width="270">

#### Further details:
- ![Presentation Slide](https://github.com/daniele21/Anomaly_Detection/blob/master/Slides.pdf)
