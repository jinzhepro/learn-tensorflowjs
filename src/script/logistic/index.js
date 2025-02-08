import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getData} from "./data";


/**
 * 将数据转换为张量并渲染散点图
 *
 * 此函数接受一个数据数组，将其渲染为散点图，并将其转换为张量用于机器学习模型的训练
 * 数据数组应包含具有 'x'、'y' 和 'label' 属性的对象，其中 'label' 可以是 0 或 1
 *
 * @param {Array} data - 包含数据对象的数组，每个对象都有 'x'、'y' 和 'label' 属性
 * @returns {Object} 返回一个包含输入张量和标签张量的对象
 */
const covertToTensor = (data)=>{
  // 渲染初始数据
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

  // 创建一个包含两个数组的数组，用于存储输入和输出数据
  return tf.tidy(()=>{
    // 打乱数据以确保随机性，有助于模型训练
    tf.util.shuffle(data)

    // 创建输入张量，包含所有数据点的 'x' 和 'y' 值
    const inputsTensor = tf.tensor(data.map(d => [d.x, d.y]))
    // 创建标签张量，包含所有数据点的 'label' 值
    const labelsTensor = tf.tensor(data.map(d => d.label))

    // 返回包含输入和标签张量的对象
    return {
      inputs: inputsTensor,
      labels: labelsTensor
    }
  })
}


/**
 * 创建一个TensorFlow.js模型
 *
 * 本函数使用TensorFlow.js库构建了一个简单的神经网络模型该模型是一个顺序模型，
 * 包含一个全连接层（Dense层）使用这种结构是因为需要对输入数据进行非线性变换，
 * 以解决特定的机器学习问题在这个例子中，模型的结构相对简单，仅用于演示目的
 *
 * @returns {tf.Sequential} 返回一个TensorFlow.js顺序模型实例
 */
const creatModel = ()=>{
  // 初始化一个顺序模型实例
  const model = tf.sequential()

  // 添加一个全连接层到模型中
  // 使用'sigmoid'激活函数，输出范围在0到1之间，适用于二分类问题
  // 本层包含一个神经元，输入形状为二维
  model.add(tf.layers.dense({activation: 'sigmoid', units: 1, inputShape: [2]}))

  // 返回构建好的模型实例
  return model
}


/**
 * 异步函数用于训练模型
 *
 * @param {Object} model - 待训练的模型实例
 * @param {tf.Tensor} inputs - 输入数据，通常是一个二维张量
 * @param {tf.Tensor} labels - 输入数据对应的标签，用于监督学习
 * @returns {Promise<tf.History>} 返回一个Promise，解析为训练历史记录的对象，包含训练过程的损失值等信息
 */
const trainModel = async (model, inputs, labels)=>{
  // 编译模型，指定损失函数和优化器
  model.compile({loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

  // 训练模型，并返回训练历史记录
  // 使用tfvis库显示训练过程的损失值
  return await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })
}


/**
 * 为给定的模型创建一个测试函数
 * 这个函数将model参数传递给一个内部函数，用于后续的预测
 * @param {Object} model - 一个包含预测方法的模型对象
 */
const testModel = (model)=>{
  /**
   * 预测函数，用于处理表单提交的预测请求
   * 此函数利用传入的model对象的predict方法，根据表单输入进行预测
   * @param {HTMLFormElement} form - 提交预测请求的表单元素
   */
  window.predict = (form)=>{
    // 使用模型的predict方法对表单输入的x和y值进行预测
    // 将表单输入转换为张量，并传递给模型的predict方法
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
    // 弹出预测结果，使用dataSync方法同步获取张量数据
    alert(pred.dataSync()[0])
  }
}

/**
 * 异步函数，用于执行物流回归分析的完整流程
 * 该函数从数据获取开始，经过模型创建、训练到测试，完成了机器学习任务的全过程
 */
async function runLogisticRegression() {
  // 从数据源获取输入数据和标签，并将其转换为张量格式，这里假设getData函数获取原始数据，covertToTensor函数将这些数据转换为张量
  const {inputs, labels} = covertToTensor(getData(400))
  // 创建物流回归模型，这里假设creatModel函数负责构建模型结构
  const model = creatModel()
  // 训练模型，这里假设trainModel函数实现了模型的训练逻辑，包括前向传播、损失计算和反向传播等步骤
  await trainModel(model, inputs, labels)
  // 测试训练好的模型，这里假设testModel函数对模型的性能进行评估
  await testModel(model)
}



window.onload = ()=>{
  runLogisticRegression().then(() => console.log('done'))
}
