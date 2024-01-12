import * as tf from '@tensorflow/tfjs-node';
import { loadMNIST } from './load-mnist-data';
import path from 'path'

const savePath = path.join(__dirname, '..', 'mnist-model.keras')
const loadPath = path.join(__dirname, '..', 'mnist-model.keras', 'model.json')

loadMNIST().then(async mnistData => {
  // Convert MNIST data into TensorFlow.js tensors
  const x_train = tf.tensor2d(mnistData.train_images.map(image => image.pixels), [mnistData.train_images.length, 784]);
  const y_train = tf.oneHot(mnistData.train_labels.map(label => label.label), 10);

  const x_test = tf.tensor2d(mnistData.test_images.map(image => image.pixels), [mnistData.test_images.length, 784]);
  const y_test = tf.oneHot(mnistData.test_labels.map(label => label.label), 10);

  // Define the model
  // const model = tf.sequential({
  //   layers: [
  //     tf.layers.inputLayer({ inputShape: [784] }),
  //     tf.layers.dense({ units: 128, activation: 'relu' }), // Hidden layer with 128 neurons and ReLU activation
  //     tf.layers.dense({ units: 10, activation: 'softmax' }), // Output layer with 10 neurons for 10 classes (digits 0-9)
  //   ],
  // });

  // // Compile the model
  // model.compile({
  //   optimizer: 'adam',
  //   loss: 'categoricalCrossentropy',
  //   metrics: ['accuracy'],
  // });
  const model = await tf.loadLayersModel(`file://${loadPath}`)
  
  // Train the model
  model
    .fit(x_train, y_train, {
      epochs: 5,
      batchSize: 32,
      validationData: [x_test, y_test],
    })
    .then(info => {
      console.log('Final accuracy', info.history.acc);
      // Evaluate the model
      const result = model.evaluate(x_test, y_test) as [tf.Scalar, tf.Scalar];
      const testLoss = result[0].dataSync()[0];
      const testAcc = result[1].dataSync()[0];
      console.log(`Test loss: ${testLoss}`);
      console.log(`Test accuracy: ${testAcc}`);
      model.save(`file://${savePath}`)
    });
});
