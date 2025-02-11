import * as overfit from "./data";
import * as xor from "../xor/data";
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

/**
 * 创建一个机器学习模型
 *
 * @param {string} tabName - 用于确定模型结构的标签名称
 * @returns {tf.Sequential} 返回一个Sequential模型
 */
/**
 * 根据不同的标签页创建不同结构的神经网络模型
 * @param {string} tabName - 当前标签页的名称，用于决定模型的结构
 * @returns {tf.Sequential} 返回一个构建好的tf.Sequential模型实例
 */
const createModel = (tabName)=>{
  // 创建一个Sequential模型实例
  const model = tf.sequential()

  // 根据tabName决定模型的层数和结构
  if(tabName === '过拟合'){
    // 当标签页为'过拟合'时，创建一个包含多层的模型以演示过拟合的处理方法

    // 添加一个全连接层，使用tanh激活函数和L2正则化，以防止过拟合
    // 这里注释掉了L2正则化配置，可能是为了简化模型或进行调试
    model.add(tf.layers.dense({
      units: 10,
      activation:"tanh",
      inputShape: [2],
      // kernelRegularizer: tf.regularizers.l2({l2: 1})
    }))

    // 添加一个Dropout层，以随机失活的方式防止过拟合
    // model.add(tf.layers.dropout({rate: 0.8}))

    // 添加一个输出层，使用sigmoid激活函数，适用于二分类问题
    model.add(tf.layers.dense({
      units: 1,
      activation: "sigmoid"
    }))
  }else{
    // 如果tabName不是'过拟合'，则创建一个简单的单层模型
    // 该层既是输入层也是输出层，使用sigmoid激活函数
    model.add(tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      inputShape: [2]
    }))
  }

  // 返回构建好的模型
  return model
}



/**
 * 异步训练模型函数
 *
 * 该函数负责编译和训练提供的模型使用给定的输入和标签张量
 * 它配置了模型的损失函数和优化器，并在训练过程中显示损失和验证损失
 *
 * @param {Object} model - 要训练的模型实例
 * @param {tf.Tensor} inputsTensor - 输入数据的张量
 * @param {tf.Tensor} labelsTensor - 目标标签的张量
 * @param {String} tabName - 在可视化选项卡中显示的名称
 * @returns {Promise<tf.History>} 训练历史记录的Promise对象，包含训练过程中的损失和验证损失
 */
const trainModel = async (model, inputsTensor, labelsTensor, tabName) => {
  // 编译模型，配置损失函数和优化器
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  })

  // 训练模型，并在指定的可视化选项卡中显示训练过程的损失和验证损失
  return model.fit(inputsTensor, labelsTensor, {
    validationSplit: 0.2,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程', tab: tabName}, ['loss', 'val_loss'], {
      callbacks: ['onEpochEnd']
    })
  })
}


/**
 * 将数据转换为张量并渲染散点图
 *
 * 此函数接受一个数据数组和一个标签参数，将数据转换为张量，并使用tfvis库渲染一个散点图
 * 数据数组应包含具有'label'，'x'和'y'属性的对象，用于分类和图表渲染
 *
 * @param {Array} data - 包含数据对象的数组，每个对象应有'label', 'x', 和'y'属性
 * @param {string} tabName - 用于tfvis渲染图表的标签页名称
 * @returns {Object} 返回一个包含输入张量和标签张量的对象
 */
const covertTensor = (data, tabName)=>{
  // 渲染散点图，展示数据分布
  tfvis.render.scatterplot({
    name: "训练数据",
    tab: tabName
  }, {
    values: [
      data.filter(d => d.label === 0),
      data.filter(d => d.label === 1)
    ],
    series: ['0', '1']
  })

  // 提取输入特征，转换为张量所需的格式
  const inputs = data.map(d => [d.x, d.y])

  // 提取标签，用于训练时与输入特征关联
  const labels = data.map(d => d.label)

  // 将输入特征和标签转换为张量
  const inputsTensor = tf.tensor(inputs)
  const labelsTensor = tf.tensor(labels)

  // 返回包含输入张量和标签张量的对象
  return {
    inputsTensor,
    labelsTensor
  }
}


/**
 * 异步执行欠拟合训练过程
 * 该函数演示了如何通过少量数据训练一个模型，以导致欠拟合的情况
 * 欠拟合发生在模型未能捕捉到数据的复杂性时，通常因为数据量不足或模型过于简单
 */
const runUnderfit = async () => {
  // 获取用于训练的欠拟合数据集
  const data = xor.getData(200);

  // 将数据转换为TensorFlow张量，适用于机器学习模型训练
  // 这里特别指明是'欠拟合'情况下的转换
  const { inputsTensor, labelsTensor } = covertTensor(data, '欠拟合')

  // 创建一个神经网络模型
  // 此函数创建并返回一个未训练的模型实例
  const model = createModel()

  // 训练模型使用转换后的张量数据
  // 这是一个异步过程，因为它可能需要时间，特别是对于大型数据集或复杂模型
  // 训练时指定模式为'欠拟合'，可能影响训练参数或输出
  await trainModel(model, inputsTensor, labelsTensor, '欠拟合')
}

/**
 * 异步函数用于演示过拟合的过程
 * 过拟合是指模型在特定数据集上训练过度，以至于模型在新数据上的泛化能力下降
 * 此函数获取数据，将其转换为张量，创建模型，并进行训练，以演示过拟合现象
 */
const runOverfit = async () => {
  // 获取用于过拟合演示的数据集
  const data = overfit.getData(200, 2);

  // 将获取的数据转换为输入张量和标签张量，标记为'过拟合'
  const { inputsTensor, labelsTensor } = covertTensor(data, '过拟合')

  // 创建一个标记为'过拟合'的模型
  const model = createModel('过拟合')

  // 使用转换后的张量数据训练模型
  await trainModel(model, inputsTensor, labelsTensor, '过拟合')

  // 打印原始数据以便进一步分析或调试
  console.log(data)
}

window.onload = ()=>{
  runUnderfit().then(() => console.log('done'))
  runOverfit().then(() => console.log('done'))
}
