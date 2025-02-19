# Train Resnet18 on Cifar10

## Model Sturcture
Based on pytorch.resnet18:
1. Change the first&last layer(conv1&fc) to match the pic size of Cifar10
2. Remove the pooling layer

## Result
model weight file is in ./saved/

(100 epoch results)

Accuracy: 0.9289

F1: 0.9289