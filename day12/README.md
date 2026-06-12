# Neural Network

## Artificial Neuron (NN Node)
Input -> Weight -> Activation -> Output
- Input: the data fed into the neuron
- Weight: importance assigned to each input
- Sum: weighted inputs are added together
- Activation: a function decides the output
- Output: passed to the next layer of neurons

## The Perceptron & Key Terms
A Perceptron is the simplest neural network<br>
Flow: Input -> Weight -> Sum -> Activation -> Output
- Weight: importance given to each input
- Bias: an additional adjustment value added to the weighted sum
- Activation Function: helps the function learn complex non-linear patterns

## Forward Propagation & Loss Function
Data moves through the network in one direction - this is the prediction phase<br>
Input -> Hidden Layer -> Output

## Loss Function
Measures the error between the network's prediction and the actual correct answer<br>
Example:
- Prediction: 90
- Actual: 100
- Error: 10

## Backpropagation & Gradient Descent
This is how learning happens - the most imp concept
- Backpropagation
    - Find error
    - Send error backward
    - Update weights
    - Repeat thousands of times
- Gradient Descent
    - Goal: Reduce error step by step

## Deep Learing Architectures
- DNN: Deep Neural Network
    - Tabular Data
- CNN: Convolution Neraul Network
    - Images
- RNN: Recurrent Neural Network
    - Text Prediction
    - Speech Recognition
- LSTM: Long Short-Term Memory
    - Chatbots
    - Language Translation
- Transformers
    - NLP
    - AI Assistants (ChatGPT, Gemini, Claude)
    - Text Generation

## Frameworks
- TensorFlow
    - Create by: Google
    - For Deployment
- Keras
    - Built on top of: TensorFlow
    - For Learning
- PyTorch
    - Create by: Meta
    - For Research