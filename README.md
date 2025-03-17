# DA6401 Assignment 1

## Author: P. Shahista Afreen

### Instructions to Train and Evaluate the Neural Network Model:

1. Install the required dependencies using:

   ```
   pip install -r requirements.txt
   ```

2. To train a neural network model for image classification on the Fashion-MNIST dataset, use the notebook **DL_A1.ipynb**.
   
   - The notebook contains the complete implementation of the neural network training framework, including model initialization, forward propagation, backpropagation, and optimization routines.
   - It supports various hyperparameter configurations and training strategies, allowing flexibility in experimentation.

3. The model is trained using different optimizers and loss functions, with options for hyperparameter tuning. The available hyperparameters include:
   - **Learning Rate**
   - **Activation Function** (tanh, sigmoid, ReLU)
   - **Initializer** (random_normal, xavier)
   - **Optimizer** (sgd, momentum, nesterov, RMSprop, Adam, nadam)
   - **Batch Size**
   - **Loss Function** (Categorical Crossentropy, Mean Squared Error)
   - **Epochs**
   - **L2 Regularization (L2_lambda)**
   - **Number of Hidden Layers and Neurons per Layer**

4. To evaluate the model performance, the notebook includes:
   - Training accuracy and loss visualization
   - Test accuracy evaluation
   - Confusion matrix analysis for better insight into model performance

5. The repository also contains images for the test and train confusion matrices, which are uploaded as `trainmatrix.png` and `testmatrix.png`.

### Explanation of the Project:

This repository contains the implementation for Assignment 1 of DA6401. The code is structured procedurally, avoiding deep learning libraries like Keras for clarity and understanding. The neural network framework supports multiple optimizers, activation functions, and loss functions, providing an end-to-end pipeline for training and evaluating models on the Fashion-MNIST dataset.

The core functions implemented in **DL_A1.ipynb** include:

#### 1. `NN_fit()`
   - Trains a neural network with specified hyperparameters.
   - Supports various optimization algorithms and loss functions.
   - Returns trained model parameters and epoch-wise cost values.

#### 2. `predict_nn()`
   - Performs a forward pass to predict output labels given input data.

#### 3. `evaluate_nn()`
   - Computes and prints training accuracy, test accuracy, and the classification report.

### Code Organization
- **DL_A1.ipynb** – The main notebook containing the complete neural network implementation, including model training, evaluation, and visualization.
- **DL_A1_Q1.ipynb** & **DL_A1_Q2.ipynb** – Initial drafts used for developing the neural network model.
- **trainmatrix.png** & **testmatrix.png** – Confusion matrix images for analyzing model performance.
- **requirements.txt** - Requirements

### Summary:
- The notebook **DL_A1.ipynb** contains the full implementation and training pipeline.
- The repository includes confusion matrix images for visualization.
- The model allows flexibility in hyperparameter selection for experimentation.

This README provides all necessary information to run the code and understand its functionality effectively.

