# Deep Learning
A subset of Machine Learning that uses Artificial Neural Networks with many layers to learn patterns from data automatically.

## Real-World Applications
- Image Recognition
- Speech Recognition
- Chatbots & NLP
- Language Translation
- Self-Driving Cars

### Weight (W)
- Represents the importance of an input feature.
- Controls how much influence an input has on the prediction.

### Bias (b)
- An extra value added to the weighted sum to help the model fit data better.
- Allows shifting the output curve up or down.

```
Neuron Output = Σ(Weight x Input) + Bias -> Activation Function -> Prediction
```

## Activation Functions
- Sigmoid
- ReLU
- Tanh
- Softmax

# Vanishing Gradient & Weight INnitialisation

## Vanishing Gradient Problem
During backpropagation in deep networks, gradients get multiplied repeatedly. With sigmoid/tanh, gradients are less than 1, so multiplying 50 times gives a number ≈ 0.

## Weight Initialisation Techniques
- Zero Init
    - W = 0
    - All neurons identical - symmetry problem
    - Never use
- Random Init
    - W - N(0, 0.01)
    - Breaks symmetry
    - May cause vanishing/exploding
- Xavier/Glorot
    - W - N(0, 2/(n<sub>in</sub>, n<sub>out</sub>))
    - Best for sigmoid/tanh
    - Most common default
- He/Kaiming
    - W - N(0, 2/n<sub>in</sub>)
    - Best for ReLU networks
    - PyTorch default

## Dropout Layer
A regularisation technique where random neurons are temporily switched off (set to 0) during each traning batch.