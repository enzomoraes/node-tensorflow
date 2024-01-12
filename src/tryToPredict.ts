import * as tf from '@tensorflow/tfjs-node';
import path from 'path'

export async function tryToPredict(pixels: number[]): Promise<tf.Tensor<tf.Rank>> {
  const savePath = path.join(__dirname, '..', 'mnist-model.keras', 'model.json')

  const loadedModel = await tf.loadLayersModel(`file://${savePath}`);
  
  const x_test = tf.reshape(pixels, [1, 784]);
  const prediction = loadedModel.predict(x_test) as tf.Tensor<tf.Rank> 
  return prediction;

}
