import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import generateBinaryClassificationDataset from "./data";

/**
 * 异步函数，用于执行逻辑回归训练和预测
 * 该函数生成一个二分类数据集，使用TensorFlow.js构建一个逻辑回归模型，训练模型，并展示训练过程
 * 最后，添加一个全局预测函数，用于根据模型预测新数据的标签
 */
async function runLogisticRegression() {
  // 生成一个包含400个样本的二分类数据集
  const data = generateBinaryClassificationDataset(400)

  // 将生成的数据集渲染为散点图，以便可视化数据分布
  await tfvis.render.scatterplot({
    name: '逻辑回归训练数据'
  },{
    values: [
      data.filter(d => d.label === 0),
      data.filter(d => d.label === 1)
    ],
    series: ['0', '1']
  })

  // 创建一个TensorFlow.js的顺序模型
  const model = tf.sequential()

  // 添加一个全连接层，使用sigmoid激活函数，输入为2维，输出为1维
  model.add(tf.layers.dense({activation: 'sigmoid', units: 1, inputShape: [2]}))

  // 编译模型，使用逻辑损失函数和Adam优化器
  model.compile({loss: tf.losses.logLoss, optimizer:tf.train.adam(0.1)})

  // 将数据集转换为Tensor格式，准备输入模型进行训练
  const input = tf.tensor(data.map(d => [d.x, d.y]))
  const output = tf.tensor(data.map(d => d.label))

  // 使用定义的参数训练模型，并展示训练过程中的损失变化
  await model.fit(input, output, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })

  // 定义一个全局预测函数，根据输入的表单数据预测标签
  window.predict = (form)=>{
    console.log(form)
    // 使用训练好的模型进行预测，并弹出预测结果
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
    alert(pred.dataSync()[0])
  }
}


window.onload = ()=>{
  runLogisticRegression().then(() => console.log('done'))
}
