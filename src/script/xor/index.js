import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from "./data";


/**
 * 异步函数，用于执行XOR回归分析
 * 包括数据准备、模型构建、训练和可视化
 */
async function runXorRegression() {
  // 获取400个数据点用于训练
  const data = getData(400)

  // 渲染散点图，可视化训练数据
  await tfvis.render.scatterplot({
    name: 'xor训练数据'
  },{
    values: [
      data.filter(d => d.label === 0),
      data.filter(d => d.label === 1)
    ],
    series: ['0', '1']
  })

  // 创建一个顺序模型
  const model = tf.sequential()

  // 添加一个具有ReLU激活函数和4个单元的密集层
  model.add(tf.layers.dense({activation: 'relu', units: 4, inputShape: [2]}))

  // 添加一个具有sigmoid激活函数和1个单元的密集层
  model.add(tf.layers.dense({activation: 'sigmoid', units: 1}))

  // 编译模型，使用逻辑损失和Adam优化器
  model.compile({loss: tf.losses.logLoss, optimizer:tf.train.adam(0.1)})

  // 将输入数据转换为Tensor
  const input = tf.tensor(data.map(d => [d.x, d.y]))
  // 将输出数据转换为Tensor
  const output = tf.tensor(data.map(d => d.label))

  // 训练模型，并在训练过程中显示损失
  await model.fit(input, output, {
    batchSize: 40,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })

  // 定义一个预测函数，用于对新的输入进行预测
  window.predict = (form)=>{
    console.log(form)
    // 使用训练好的模型进行预测
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
    // 输出预测结果
    alert(pred.dataSync()[0])
  }
}



window.onload = ()=>{
  runXorRegression().then(() => console.log('done'))
}
