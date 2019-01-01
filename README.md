# Simple-CLF-GAN
MNIST generator and classifier</br>
Generative Adversarial Network generates MNIST digits using 2 simple feed forward neural networks for discriminator and generator. Each digit is then classified by basic (1 hidden layer) MNIST classifier.

## Examples of generated images:
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/examples.png)

## How to run?
In order to run this script, you will need the following dependencies:

- TensorFlow
- SciPy
- Matplotlib

Then, just run the script with python:

    python gan_and_clf.py

The script will automatically start training the model. For every 500 steps, new image will be generated and saved to `output` folder.
