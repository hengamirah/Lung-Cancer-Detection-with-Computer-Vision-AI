Lung cancer is the leading cause of cancer-related deaths worldwide, largely due to late-stage diagnoses. Early detection significantly improves survival rates, with advanced machine learning methods offering potential for accurate and timely diagnoses. This study applies four deep learning models (ResNet50, GoogLeNet, MobileNet, and YOLOv11) to classify histopathological lung images from the LC25000 dataset into three categories: benign tissue, lung adenocarcinoma, and squamous cell carcinoma. The dataset, comprising 15,000 images evenly distributed across the three classes, was preprocessed using resizing, augmentation, and stratified splitting techniques. Model performance was evaluated using metrics such as precision, recall, and F1-score. Results indicate that YOLOv11 outperformed other models with the highest precision (99.3%), recall (99.7%), and F1-score (99.5%), followed by ResNet50 with an F1-score of 98.7%. MobileNet demonstrated strong computational efficiency, while GoogLeNet achieved lower precision (75.6%) and F1-score (83.7%), highlighting areas for optimization. Misclassifications were primarily observed between adenocarcinoma and squamous cell carcinoma, likely due to histopathological similarities. The findings underscore YOLOv11’s potential for accurate lung cancer diagnosis. Future efforts will focus on expanding the dataset to include other cancer types and further enhancing model performance to address classification challenges.
Keywords: Lung cancer, histopathology, deep learning, YOLOv11, ResNet50, LC25000 dataset

RESULTS
   
1.1 ResNet (Residual Network)
ResNet performs well on balanced multi-class datasets, excels at extracting features from high-resolution images like the 768x768 pixel histopathological images in LC25000. The ImageNet pre-trained ResNet can be fine-tuned on the LC25000 dataset with minimal adjustments, leveraging its existing feature extraction capabilities. The model shows high training accuracy in the final training step 100% and validation accuracy at 99.16%. The confusion matrix shows there are a total of: True Positive- 732, True Negative-1490, False Positive-10 and False Negative-18. The classification report also shows 98.6% precision, 97.6% recall and 98.1% f1-score, which concludes a high reliable model.

![image](https://github.com/user-attachments/assets/f3ad5bad-9547-48e0-8633-af7207063fe1) ![image](https://github.com/user-attachments/assets/4ddf36e8-2028-4f94-862b-1d5950bb5ac7)


Figure 9 (Left) ResNet Model training and validation loss, (Right) training and validation accuracy for 5 epochs.
 
 ![image](https://github.com/user-attachments/assets/42438698-b625-4dfa-b478-640d07cf30da)

1.2 GoogLeNet 
The GoogLeNet model was trained for 5 epochs on the lung cancer histopathological dataset, showing the following performance metrics:
1.	Training Progress:
Initial training accuracy	58.84% 
Final training accuracy	89.87%
Initial validation accuracy	82.44%
Final validation accuracy	89.87%
2.	Learning Curve Analysis:
•	The model shows steady improvement in both training and validation metrics
•	Training loss decreased significantly from 145.3983 to 2.2715
•	Validation loss improved from 7.5353 to 2.1043
•	The curves suggest good learning with training and validation accuracy converging, indicating effective model learning without significant overfitting
•	Best validation accuracy of 90.27% achieved at epoch 3
 Figure 10 Training Progress Over 5 Epochs

![image](https://github.com/user-attachments/assets/0de0d491-de6d-475b-b49c-c487fa7f58a3)

 ![image](https://github.com/user-attachments/assets/9af71693-9446-4e0b-9930-e3da24f9c9ed)

Figure 11  Model Accuracy and Loss Curves
The accuracy plot shows steady improvement in both training and validation accuracy, while the loss plot demonstrates consistent decrease in both training and validation loss, indicating good model convergence.
3.	Confusion Matrix Analysis: 
The confusion matrix reveals the model's classification performance across three classes:
•	Lung adenocarcinoma (lung_aca): 703 correct predictions, with 14 misclassified as normal and 33 as squamous cell carcinoma
•	Normal lung tissue (lung_n): 713 correct predictions, with 37 misclassified cases.
•	Lung squamous cell carcinoma (lung_scc): 559 correct predictions, with notable misclassifications.


![image](https://github.com/user-attachments/assets/279d124c-6776-41ac-8a7c-8722dd04a5b9)

![image](https://github.com/user-attachments/assets/83588d95-8c51-47b2-880d-6754056305d2)

 
 
Figure 12 Confusion Matrix and Classification Report
1.	Performance Metrics by Class:
o	Precision scores:
o	Lung adenocarcinoma: 0.76
o	Normal lung tissue: 0.98
o	Squamous cell carcinoma: 0.94
o	Overall accuracy: 0.88
o	Macro and weighted averages: 0.89
	These results demonstrate robust model performance, with potential areas for optimization including:
o	Fine-tuning the balance between adenocarcinoma and squamous cell carcinoma classification
o	Implementing targeted data augmentation for commonly confused cases
o	Exploring architectural modifications to improve feature discrimination between cancer types
o	Considering class-specific learning strategies to address the slight performance variations between classes


1.3 YOLO V11 Model
The training and validation loss graph are plotted in Figure 4. In Table1, the top5 accuracy indicates that the correct category is predicted among the top 5 with the highest prediction probability, and top1 accuracy shows the correct category among the top 1 with the highest prediction probability.
 
Figure 13 YOLO model Training and Validation loss on lung dataset, with metrics accuracy for top1 and top5
epoch	time	Training loss	metrics/ accuracy top1	metrics/ accuracy top5	Validation loss

![image](https://github.com/user-attachments/assets/6989cb4e-bf66-4a7a-b5e2-f235ba57dd6f)
Model Training results for 10 epochs

1.	Model Performance on Test Data
 ![image](https://github.com/user-attachments/assets/cd2118f4-c849-4a9f-be19-2bf3f10d92ae)

Figure 14 Actual Vs Predicted class for 9 sample images from test data
The model conducts a prediction on unseen test data and the performance is evaluated through Classification Report (Figure 6) and Confusion Matrix (Figure 7). The classification report shows a high accuracy result with 99.3% score for precision, 99.7% recall and 99.5% f1-score with number of 748 True Positives, 1495 True Negatives, 5 False Positives and 2 False Negatives.
 
Figure 15 YOLOv11 Model Classification report
![image](https://github.com/user-attachments/assets/d5a9bfce-ad2e-4ec3-8a37-92250687509f)

 ![image](https://github.com/user-attachments/assets/dd247b3d-9e93-4d18-b9c4-2424cad6cd62)

