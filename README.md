# Simple-CLF-GAN
MNIST generator and classifier</br>
Generative Adversarial Network generates MNIST digits using 2 simple feed forward neural networks for discriminator and generator. Each digit is then classified by basic (1 hidden layer) MNIST classifier.

## Examples of generated images:
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/35000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/38000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/39500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/40000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/45500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/48500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/50500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/53000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/55000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/57500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/61000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/62500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/72000.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/74500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/75500.jpg)
![alt text](https://raw.githubusercontent.com/gstark0/Simple-CLF-GAN/master/examples/80500.jpg)

## How to run?
In order to run this script, you will need the following dependencies:

- TensorFlow
- SciPy
- Matplotlib

Then, just run the script with python:

    python gan_and_clf.py

The script will automatically start training the model. For every 500 steps, new image will be generated to 'output' folder.
