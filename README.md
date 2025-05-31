
# 🐶🐱 Dog vs Cat Image Classification using CNN & Transfer Learning

This project tackles the classic binary image classification problem of distinguishing between images of dogs and cats. It uses a deep learning approach with TensorFlow, leveraging transfer learning from a pre-trained CNN model to boost performance and reduce training time.
## 📂 Dataset

Source: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset 

Size: 25,000 labeled images (Dogs and Cats)

Preprocessing:
Corrupt images removed using PIL and TensorFlow

Train/Test split with splitfolders (80%/20%)
## 🧠 Model Architecture
Base Model: Pre-trained VGG16 or MobileNetV2 (transfer learning)
Layers:

Flatten layer

Dense layer with 128 units + ReLU

Dropout for regularization

Dense layer with 64 units + ReLU

Output layer with 1 neuron + Sigmoid (for binary classification)

Loss Function: BinaryCrossentropy

Optimizer: Adam

Metric: Accuracy
## 🎛️ Data Augmentation

Applied using ImageDataGenerator to improve generalization and reduce overfitting:

rotation_range=20

width_shift_range=0.2

height_shift_range=0.2

shear_range=0.2

zoom_range=0.2

horizontal_flip=True

rescale=1./255

This increases the robustness of the model by simulating real-world variations.
## 📈 Performance Comparison


| Model                             | Train Accuracy | Validation Accuracy | Validation Loss     |
|----------------------------------|----------------|----------------------|----------------------|
| Transfer Learning + Augmentation | 92.18%         | 94.22%               | 0.1646               |
| Plain CNN                        | 80%            | 63%                  | High (Overfitting)   |

> This highlights how augmentation and transfer learning can significantly enhance performance and reduce overfitting.

## 🚀 Features
✅ Real-world dataset with corrupted image handling

✅ Transfer learning for improved accuracy and training 
efficiency

✅ Data augmentation for better generalization

✅ Visualization of training and validation accuracy/loss

✅ Clean, modular code for easy experimentation


# 🧪 Installation


add your kaggle.json to the .kaggle/ directory:

mkdir -p ~/.kaggle

cp kaggle.json ~/.kaggle/

pip install tensorflow kaggle split-folders

## 🏁 Run Instructions

Open Dog_vs_cat_Classification.ipynb

Execute all cells sequentially:

Download and unzip dataset

Preprocess and clean corrupt images

Split data

Train model with augmentation + VGG16

Visualize performance
# 👨‍💻 Author
Prabhjot Singh  Information Technology | Marymount University AI Enthusiast | Building end-to-end ML/DS projects


# ⭐ GitHub Tags

#CNN #Classification #DeepLearning #TransferLearning #Data Augementation #Tensorflow