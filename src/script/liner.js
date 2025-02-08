import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'


const height = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
const weight = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

/**
 * 异步函数，用于执行线性回归分析
 * 该函数从输入数据（身高）预测输出数据（体重）
 */
async function runLinearRegression() {

  // 定义数据
  // 对身高数据进行归一化处理，减去150后除以10
  const input = tf.tensor(height).sub(150).div(10);
  // 对体重数据进行归一化处理，减去40后除以10
  const output = tf.tensor(weight).sub(40).div(10);

  // 定义一个简单的线性回归模型（单层，单神经元）
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // 编译模型，使用均方误差作为损失函数，随机梯度下降作为优化器
  model.compile({loss: 'meanSquaredError', optimizer: tf.train.sgd(0.01)});

  // 训练模型
  await model.fit(input, output, {
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  });

  // 使用训练好的模型对输入数据进行预测
  let prediction = model.predict(input);

  // 可视化预测结果与真实值的对比
  await tfvis.render.scatterplot({
    name: '线性回归预测结果',
  },{
    values: [
      height.map((x, i) => ({x, y: weight[i]})),
      height.map((x, i) => ({x, y: prediction.mul(10).add(40).dataSync()[i]})),
    ],
    series: ['真实值', '预测值']
  },{
    xAxisDomain: [140, 250],
    yAxisDomain: [30, 150]
  })

  // 使用训练好的模型对特定的身高值进行预测，并弹出预测的体重值
  alert(model.predict(tf.tensor([260]).sub(150).div(10)).mul(10).add(40).dataSync())
}


window.onload = () => {
  runLinearRegression().then(r => {
    console.log('done')
  });
}
