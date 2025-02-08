import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {data} from "./data";


/**
 * 将数据转换为标准化的Tensor格式
 * 此函数首先对数据进行洗牌，然后将输入（Horsepower）和标签（Miles_per_Gallon）转换为Tensor格式
 * 接着，对输入和标签进行标准化处理，以确保它们在训练模型时具有相似的尺度
 * 最后，返回包含标准化后的输入、标签及其最大和最小值的对象
 *
 * @param {Array} data - 包含Horsepower和Miles_per_Gallon属性的数据数组
 * @returns {Object} 包含标准化后的输入、标签及其最大和最小值的对象
 */
const covertTensor = (data)=>{
  // 渲染初始数据
  tfvis.render.scatterplot({
    name: '训练数据',
  }, {
      values: data.map(d => ({
        x: d.Horsepower,
        y: d.Miles_per_Gallon
      }))
    },
    {
      xLabel: '马力',
      yLabel: '每加仑英里数',
    }
  )
  // 使用tf.tidy确保在函数执行完毕后清理内存
  return tf.tidy(()=>{
    // 对数据进行洗牌，以增加数据的随机性
    tf.util.shuffle(data)

    // 将输入（Weight_in_lbs）和标签（Miles_per_Gallon）转换为2D Tensor
    const inputsTensor = tf.tensor2d(data.map(d => [d.Horsepower]))
    const labelsTensor = tf.tensor2d(data.map(d => [d.Miles_per_Gallon]))

    // 计算输入和标签的最大值和最小值，用于标准化
    const inputsMax = inputsTensor.max()
    const inputsMin = inputsTensor.min()
    const labelsMax = labelsTensor.max()
    const labelsMin = labelsTensor.min()

    // 对输入和标签进行标准化处理，将其值域限制在0到1之间
    const inputNormalized = inputsTensor.sub(inputsMin).div(inputsMax.sub(inputsMin))
    const labelsNormalized = labelsTensor.sub(labelsMin).div(labelsMax.sub(labelsMin))

    // 返回包含标准化后的输入、标签及其最大和最小值的对象
    return {
      inputs: inputNormalized,
      labels: labelsNormalized,
      inputsMax,
      inputsMin,
      labelsMax,
      labelsMin
    }
  })
}


/**
 * 创建一个简单的神经网络模型
 *
 * 该函数使用TensorFlow.js库构建了一个顺序模型，该模型由两个全连接层（Dense layers）组成
 * 第一层接收一个形状为[1]的输入，即单个特征，第二层输出一个单元，整个模型旨在进行简单的线性回归任务
 *
 * @returns {tf.Sequential} 返回一个TensorFlow.js顺序模型实例
 */
const creatModel = ()=>{
  // 创建一个顺序模型实例
  const model = tf.sequential()

  // 添加第一个全连接层，包含1个神经元，输入形状为[1]
  model.add(tf.layers.dense({units: 1, inputShape: [1]}))
  // 添加第二个全连接层，包含1个神经元，用于输出
  model.add(tf.layers.dense({units: 1}))

  // 返回构建好的模型
  return model
}


/**
 * 异步训练模型函数
 *
 * 该函数负责编译和训练传入的模型使用指定的输入数据和标签数据
 * 它使用Adam优化器和均方误差损失函数进行编译，并通过批量处理和多个时期来训练模型
 * 训练过程的损失情况将通过回调函数进行可视化
 *
 * @param {Object} model 要训练的模型对象
 * @param {Object} inputs 输入数据
 * @param {Object} labels 标签数据
 * @returns {Object} 训练结果
 */
const trainModel = async (model, inputs, labels)=>{
  // 编译模型，指定损失函数和优化器
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.1),
  })

  // 训练模型，指定批量大小、训练时期数和回调函数
  return await model.fit(inputs, labels, {
    batchSize: 32,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'])
  })
}


/**
 * 测试模型的性能并可视化预测结果
 * 该函数使用给定的模型对输入数据进行预测，并将预测结果与原始数据进行比较，通过散点图进行可视化
 *
 * @param {Object} model - 训练好的模型，用于进行预测
 * @param {Array} inputData - 原始输入数据，用于可视化
 * @param {Object} normalizedData - 包含数据归一化所需的最大值和最小值，用于将数据恢复到原始范围
 */
const testModel = (model, inputData, normalizedData)=>{
  // 从归一化数据中解构出输入和标签的最大值和最小值
  const {inputsMax, inputsMin, labelsMax, labelsMin} = normalizedData
  // 使用tf.tidy()管理内存，减少内存泄漏风险
  const [xs, preds] = tf.tidy(() => {
    // 生成0到1之间的100个等间距数值
    const xs = tf.linspace(0, 1, 100)
    // 使用模型对生成的数值进行预测
    const preds = model.predict(xs.reshape([100, 1]))

    // 将归一化的输入数据恢复到原始范围
    const unNorXs = xs.mul(inputsMax.sub(inputsMin)).add(inputsMin)
    // 将归一化的预测数据恢复到原始范围
    const unNorPreds = preds.mul(labelsMax.sub(labelsMin)).add(labelsMin)
    // 同步获取数据并返回
    return [unNorXs.dataSync(), unNorPreds.dataSync()]
  })
  // 将预测的数据点整理为[{x, y}]格式
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  })
  // 将原始输入数据整理为[{x, y}]格式，用于可视化
  const originalPoints = inputData.map(d => ({
    x: d.Horsepower,
    y: d.Miles_per_Gallon
  }))
  // 使用TensorFlow.js的可视化工具渲染散点图，比较原始数据和预测数据
  tfvis.render.scatterplot(
    {name: '线性回归模型预测结果'},
    {
      values: [
        originalPoints,
        predictedPoints
      ],
      series: ['原始数据', '预测数据']},
    {
      xLabel: '马力',
      yLabel: '每加仑英里数',
    }
  )
}

/**
 * 异步执行线性回归流程
 * 本函数负责从数据准备、模型创建、模型训练到模型测试的全过程
 */
async function runLinearRegression() {
  // 将原始数据转换为张量格式，以便后续的模型训练和测试
  const {inputs, labels} = covertTensor(data)

  // 创建线性回归模型
  const model = creatModel()

  // 训练模型，使其学会输入数据与标签之间的映射关系
  await trainModel(model, inputs, labels)

  // 测试模型，评估其在未见过的数据上的表现
  await testModel(model, data, covertTensor(data))
}

window.onload = () => {
  runLinearRegression().then(r => {
    console.log('done')
  });
}
