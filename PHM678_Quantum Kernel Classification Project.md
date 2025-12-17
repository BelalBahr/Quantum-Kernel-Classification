PHM 678 
Quantum Machine learning 
and Artificial Intelligence 
Fall 2025 
Quantum Kernel Classification Project 
Items in red are optional for extra credit 
Overview 
In this project, you will implement, analyze, and extend quantum kernel methods for machine 
learning. You will build quantum feature maps, simulate quantum kernels, and benchmark the 
resulting quantum-enhanced classifiers against classical baselines. You are also encouraged to 
experiment with feature map design, simulate noise effects, and study scaling behaviors as 
part of a deeper exploration. 
This project aims to deepen your understanding of how quantum properties can influence learning 
tasks, and where quantum advantages might arise (or fail to). 
Objectives 
By completing this project, you will: 
 Implement quantum feature maps and compute quantum kernels. 
 Understand the theoretical basis for quantum-enhanced feature spaces. 
 Build and train SVM classifiers using quantum kernels. 
 Compare quantum models with classical ones both empirically and theoretically. 
Core Tasks 
�
� Part 1: Theoretical Background 
 Review and summarize the key ideas behind quantum kernel methods: 
o What is a quantum feature map? 
o How is a quantum kernel matrix constructed? 
�
� Part 2: Implementing Quantum Feature Maps 
 Prepare a small real-word binary classification dataset (e.g. “Breast cancer Wisconsin 
dataset” from scikit-learn). 
 Implement at least two different quantum feature maps: 
1. A basic feature map (e.g., using RX, RZ rotations based on input features). 
2. A more complex entangled feature map (e.g., ZZ Feature Map, custom-designed 
map). 
 Visualize each circuit for a few input samples. 
 Discuss the intuition behind each design: what kind of entanglement or expressivity are 
you introducing? 
Deliverable: Feature map circuits, code, and visualizations. 
�
� Part 3: Quantum Kernel Computation 
For each of the two different quantum feature maps circuit 
 Simulate the circuits to compute the quantum kernel matrix for a dataset. 
 Visualize the kernel matrix as a heatmap for the first 50 entries only, comment on the plot. 
Deliverable: Code to compute kernels, heatmaps, and validation. 
�
� Part 4: Classical SVM Training and Evaluation 
 Train classical SVMs on the quantum kernels. 
 Tune hyperparameters (e.g., SVM regularization parameter C). 
 Measure and report classification accuracy. 
Compare to: 
 Classical SVMs trained with RBF, polynomial, and linear kernels. 
Deliverable: Performance results, comparison tables. 
�
� Part 5: Angle Embedding. 
 Apply Principal Component Analysis (PCA) on the dataset. 
 Implement and train a QSVM using angle encoding. 
 Measure and report classification accuracy, how does it differ than results in part 4. 
