import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from "./data";


/**
 * 将数据转换为Tensor并渲染散点图
 *
 * 此函数接受一个数据数组，将其渲染为散点图，并将其转换为Tensor格式
 * 主要用于数据预处理和可视化
 *
 * @param {Array} data - 包含数据点的数组，每个数据点包含x、y坐标和标签
 * @returns {Object} 返回一个包含输入张量(inputs)和标签张量(labels)的对象
 */
const covertToTensor = (data) => {
  // 渲染初始数据
  // 创建一个散点图，展示数据集中标签为0和1的数据点
  // 散点图的名称为'训练数据'，并根据标签将数据分为两个系列
  tfvis.render.scatterplot({
      name: '训练数据',
    }, {
      values: [
        data.filter(d => d.label === 0),
        data.filter(d => d.label === 1)
      ],
      series: ['0', '1']
    },
  )

  // 使用tf.tidy函数管理内存，减少Tensor操作产生的内存泄漏
  return tf.tidy(()=>{
    // 将数据点的x和y坐标转换为Tensor
    const inputsTensor = tf.tensor(data.map(d => [d.x, d.y]))
    // 将数据点的标签转换为Tensor
    const labelsTensor = tf.tensor(data.map(d => d.label))

    // 返回包含输入和标签Tensor的对象
    return {
      inputs: inputsTensor,
      labels: labelsTensor
    }
  })
}


/**
 * 创建一个神经网络模型
 *
 * 本函数使用TensorFlow.js库构建了一个简单的神经网络模型该模型包含两个全连接层（Dense layers）：
 * - 第一层使用ReLU激活函数，包含4个神经元，输入形状为2维
 * - 第二层使用Sigmoid激活函数，包含1个神经元，用于二分类问题
 *
 * @returns {tf.LayersModel} 返回一个TensorFlow.js的LayersModel实例
 */
const createModel = () => {
  // 创建一个顺序模型
  const model = tf.sequential()

  // 添加第一个全连接层，使用ReLU激活函数，4个神经元，输入数据为2维
  model.add(tf.layers.dense({activation: 'relu', units: 4, inputShape: [2]}))

  // 添加第二个全连接层，使用Sigmoid激活函数，1个神经元，适用于二分类问题
  model.add(tf.layers.dense({activation: 'sigmoid', units: 1}))

  // 返回构建好的模型
  return model
}


/**
 * 使用给定的模型对输入数据进行训练
 *
 * @param {tf.LayersModel} model - 待训练的模型
 * @param {tf.Tensor} inputs - 输入数据，用于模型训练
 * @param {tf.Tensor} labels - 标签数据，与输入数据对应，用于计算损失
 * @returns {Promise<tf.History>} - 返回训练历史记录的Promise对象，包含训练过程中的损失信息
 */
const trainModel = (model, inputs, labels)=>{
  // 编译模型，指定损失函数和优化器
  model.compile({loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

  // 训练模型，并使用可视化工具监控训练过程中的损失变化
  return model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })
}


/**
 * 为给定的模型创建一个测试方法
 * 此函数接收一个模型作为参数，并定义了一个预测函数，该预测函数使用模型对表单数据进行预测
 * @param {Object} model - 用于预测的模型对象
 */
const testModel = (model)=>{
  /**
   * 预测函数
   * 此函数从表单中提取数据，使用模型进行预测，并显示预测结果
   * @param {Object} form - 包含预测所需数据的表单对象
   */
  window.predict = (form)=>{
    // 输出表单数据以供调试
    console.log(form)
    // 使用模型对表单数据进行预测
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
    // 显示预测结果
    alert(pred.dataSync()[0])
  }
}



/**
 * 异步函数用于执行异或回归任务
 * 该函数从数据获取开始，到模型训练，最后到模型测试的全过程
 */
async function runXorRegression() {
  // 将原始数据转换为张量格式，适合深度学习模型处理
  const {inputs, labels} = covertToTensor(getData(400))

  // 创建一个深度学习模型，用于后续的训练
  const model = createModel()

  // 使用输入和标签数据训练模型
  await trainModel(model, inputs, labels)

  // 训练完成后，测试模型性能
  await testModel(model)
}




window.onload = ()=>{
  runXorRegression().then(() => console.log('done'))
}
