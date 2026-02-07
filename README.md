# 📚 Project Index

This repository covers:

- Parallel Computing using modern ISO C++ and HPC concepts  
- Deep Learning fundamentals and hands‑on exercises  
- Artificial Intelligence foundations  
- Convolutional Neural Networks and Computer Vision  
- PyTorch and PyTorch Lightning workflows  
- Generative AI and GAN implementations  
- Edge‑AI and Jetson setup notes  

---

# 🎯 Project Goal

- Build and run Deep Learning networks on a High‑Performance Computing (HPC) platform from scratch.  
- Requires strong knowledge of **C++**, **Python**, and **parallel programming**.  
- This repository acts as a **reference notebook**—concise, practical, and intended for readers who already understand the underlying theory.

---

# 🛠 Requirements

- A powerful PC and an Edge Computing Platform (e.g., NVIDIA Jetson REStudio J4011)  
- Familiarity with C++, Python, and parallel programming  

---

# 📦 Project Structure

The project is divided into **Modules**, **Sub‑chapters**, and **Exercises**.

---

## **Module 1: HPC & Parallelism**

Focus: ISO C++ parallelism, GPU acceleration, and HPC fundamentals.  
(Lessons are intentionally unordered in this module.)

### Files Included
- [DAXPY.md](./DAXPY.md)  
- [ISO_CPP_Algorithms_HPC_Documentation.md](./ISO_CPP_Algorithms_HPC_Documentation.md)  
- [Indexing_in_Parallel_Computing.md](./Indexing_in_Parallel_Computing.md)  
- [NVIDIA_Grace_Hopper_Coherent_HW.md](./NVIDIA_Grace_Hopper_Coherent_HW.md)  
- [Parallel_Algorithms_CPP.md](./Parallel_Algorithms_CPP.md)  
- [deep_learning_excercise.md](./deep_learning_excercise.md)  
- [excercises.cpp](./excercises.cpp)  
- [excercises_1.cpp](./excercises_1.cpp)  
- [excercises_2.cpp](./excercises_2.cpp)  
- [excercises_3.cpp](./excercises_3.cpp)  
- [extra_excercise_1.cpp](./extra_excercise_1.cpp)  
- [introduction.md](./introduction.md)  
- [main.cpp](./main.cpp)  

---

## **Module 2: Deep Learning Introduction**

Foundational concepts and early‑stage neural network understanding.

### Files Included
- [deepLearningIntroduction.md](./deepLearningIntroduction.md)  
- [neurel_network_architecture.md](./neurel_network_architecture.md)  
- [neural_network_basics.md](./neural_network_basics.md)  
- [neural_network_introduction.md](./neural_network_introduction.md)  
- [Bias_Variance_Tradeoff.ipynb](./Bias_Variance_Tradeoff.ipynb)  

---

## **Module 3: Convolutional Neural Networks**

CNN fundamentals, preprocessing, augmentation, and full implementations.

### Files Included
- [convolutional_neural_networks.md](./convolutional_neural_networks.md)  
- [cnn_architecture.md](./cnn_architecture.md)  
- [cnn_with_python.md](./cnn_with_python.md)  
- [cnn_cifar10_full.py](./cnn_cifar10_full.py)  
- [computer_vision_basics.md](./computer_vision_basics.md)  
- [preprocessing_image_dataset.py](./preprocessing_image_dataset.py)  
- [image_augmentation.py](./image_augmentation.py)  
- [convolve_and_pool.py](./convolve_and_pool.py)  

---

## **Module 4: Advanced Neural Network Topics (including AI Foundations)**

Training, optimization, MLPs, RNNs, Transformers, classical ML preprocessing, and more.

### Files Included
- [training_a_neural_network.md](./training_a_neural_network.md)  
- [classify_news_article.py](./classify_news_article.py)  
- [multilayer_perceptron.md](./multilayer_perceptron.md)  
- [multilayer_perceptron.py](./multilayer_perceptron.py)  
- [recurrent_neural_network.md](./recurrent_neural_network.md)  
- [transformer_architecture.md](./transformer_architecture.md)  
- [types_of_neural_network.md](./types_of_neural_network.md)  
- [lossfunction_for_dl.md](./lossfunction_for_dl.md)  
- [lossfunction_optimization_algo.md](./lossfunction_optimization_algo.md)  
- [hyperparameter_tuning.md](./hyperparameter_tuning.md)  
- [optimizing_deeplearning_mdl.md](./optimizing_deeplearning_mdl.md)  
- [L1_Regularization.md](./L1_Regularization.md)  
- [L2_regularization.mde](./L2_regularization.mde)  
- [elastic_dropout_regularization.md](./elastic_dropout_regularization.md)  
- [wine.csv](./wine.csv)  
- [wine_preprocessing_example.py](./wine_preprocessing_example.py)  

---

## **Module 5: Build Neural Networks with PyTorch**

Pure PyTorch → PyTorch Lightning progression with classification and regression demos.

### Files Included
- [build_nn_with_PyTorch.md](./build_nn_with_PyTorch.md)  
- [pytorch_lightning_demo.py](./pytorch_lightning_demo.py)  
- [pytorch_lightning_classification.py](./pytorch_lightning_classification.py)  
- [pytorch_lightning_regression.py](./pytorch_lightning_regression.py)  
- [run_hparam_tuning.py](./run_hparam_tuning.py)  

---

## **Module 6: Generative AI & GANs**

GAN theory, implementation, and experiments with anime faces, Fashion‑MNIST, and text.

### Files Included
- [GANs.md](./GANs.md)  
- [GANs_in_detail.md](./GANs_in_detail.md)  
- [GAN_generator.md](./GAN_generator.md)  
- [GAN_dcgan_anime.py](./GAN_dcgan_anime.py)  
- [GAN_anime_real_vs_bad_fake.py](./GAN_anime_real_vs_bad_fake.py)  
- [GAN_anime_real_vs_good_fake.py](./GAN_anime_real_vs_good_fake.py)  
- [gan_customer_reviews.py](./gan_customer_reviews.py)  
- [train_gan_fashion_mnist.py](./train_gan_fashion_mnist.py)  
- [generative_ai.md](./generative_ai.md)  

---

## **Module 7: Edge AI & Jetson**

Setup and deployment notes for edge‑AI platforms.

### Files Included
- [setting_up_jetson.md](./setting_up_jetson.md)  

---

# 👤 Current Status

All project activities are currently done individually with GPT‑4 assistance.  
Once the base version stabilizes, collaborators will be invited to expand the work.

---

# ▶️ Python Example

```bash
pip install torch pandas scikit-learn
python wine_preprocessing_example.py
python classify_news_article.py
