
The code in this repo contains my solution approach for kaggle competition: https://www.kaggle.com/c/humpback-whale-identification

The task is openset fine-grained recognition. We need to identify humpback whale (out of 5004 “classes”) by observing image of its tail, or say “it is unknown one”.
Some of the initial approaches followed are: 
1) Image classification task: Even though the number of classes for classification is large, we are trying some state of art ResNet 50 and ResNet 101 architecure. We are trying some data augmentation to increase the number of images per class. Image classification task is giving map score of around 0.80 ( after ensembling).
2) Siamese Network with hard negative mining of image pairs: Here I have referred to solution approach of Martin Piotte for playground competition https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563/output. Siamese network are neural networks containing two or more identical subnetwork components. The main idea behind siamese networks is that they can learn useful data descriptors of different whale classes , even with low number of images and that can be used to determine whether a new image belong to existing 5004 classes of images or it is a new whale category (not seen before hand). This network is very popular for one shot learning task. 
Also as we need to provide positive and negative image pairs for training, we are using 'Linear sum assignment' algorithm to find most difficult matching image pairs.Simaese network is providing 0.87 map score after training for 300 epochs. 





