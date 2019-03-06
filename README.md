#Humpback Whale Identification Challenge

The code in this repo contains my solution approach for kaggle competition: https://www.kaggle.com/c/humpback-whale-identification

The task is openset fine-grained recognition. We need to identify humpback whale (out of 5004 “classes”) by observing image of its tail, or say “it is unknown one”.
I have followed following appraoches for this problem. 
1) Image classification task: Here we train  ResNet 50 and ResNet 101 architecure on 5004 classes. Once we have trained the network, new whale images are inserted in validation set (1.3X of original validation data). We determine the optimal softmax score for for new whale catergory by bruteforce search in range 0 to 0.95. 
For test data, we create probabilty score for 5004 classes using ResNet softmax probabilities and for new_whale category we use the optimal score determined using validation set. 

2) Siamese Network with hard negative mining of image pairs: Here I have referred to solution approach of Martin Piotte  https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563/output . I have tried to implement this solution using pytorch. 




