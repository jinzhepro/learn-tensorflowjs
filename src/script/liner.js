import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const x = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
const y = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

window.onload = async function() {
  tfvis.render.scatterplot({
    name: '线性回归'
  }, {
    values: x.map((x, i) => ({x, y: y[i]}))
  },{
    xAxisDomain: [140, 250],
    yAxisDomain: [30, 140]
  })



  const model = tf.sequential()

  model.add(tf.layers.dense({units: 1, inputShape: [1]}))
  model.compile({loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.01)})

  const inputs = tf.tensor(x).sub(150).div(10)
  const labels = tf.tensor(y).sub(40).div(10)

  await model.fit(inputs, labels, {
    batchSize: 10, // 使用几个样本训练梯度
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })

  const output = model.predict(tf.tensor([300]).sub(150).div(10))
  alert(output.mul(10).add(40).dataSync()[0])
}
