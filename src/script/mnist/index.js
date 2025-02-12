import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {MnistData} from "./data/data";


/**
 * 创建并返回一个用于图像分类的深度学习模型
 * 该模型使用了卷积神经网络结构，适合处理28x28大小的灰度图像
 * 主要用于MNIST数据集的数字识别等任务
 *
 * @returns {tf.Sequential} 一个Sequential模型实例，包含了整个网络结构
 */
const createModel = ()=>{
  // 创建一个Sequential模型实例
  const model = tf.sequential()

  // 添加第一个卷积层
  // 输入形状为28x28x1，适用于单通道（灰度）图像
  // 使用5x5的卷积核，8个过滤器，步长为1
  // 激活函数为ReLU，权重初始化使用方差缩放
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling',
  }))

  // 添加最大池化层
  // 使用2x2的池化窗口，步长为2，用于下采样
  model.add(tf.layers.maxPool2d({poolSize: [2, 2], strides: [2, 2]}))

  // 添加第二个卷积层
  // 保持5x5的卷积核，增加到16个过滤器，步长为1
  // 激活函数为ReLU，权重初始化使用方差缩放
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling',
  }))

  // 再次添加最大池化层，进行下采样
  model.add(tf.layers.maxPool2d({poolSize: [2, 2], strides: [2, 2]}))

  // 添加扁平化层，将三维输出展平为一维
  model.add(tf.layers.flatten())

  // 添加全连接层（密集层）
  // 输出10个单位，对应0-9共10个类别
  // 激活函数为softmax，适用于多分类问题
  // 权重初始化使用方差缩放
  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling',
  }))

  // 返回构建好的模型
  return model
}


/**
 * 异步函数用于准备和转换MNIST数据集
 * 此函数首先创建一个MnistData实例，然后加载数据集
 * 接着，它从数据集中提取训练数据和测试数据，并将它们格式化为适合训练的格式
 * 最后，函数返回一个对象，包含格式化后的训练数据和测试数据
 */
const covertData = async () => {
  // 创建一个新的MnistData实例
  const data = new MnistData()
  // 加载MNIST数据集
  await data.load()
  // 从数据集中获取1000个训练样本
  const d = data.nextTrainBatch(1000)
  // 从数据集中获取200个测试样本
  const t = data.nextTrainBatch(200)
  // 将训练样本的输入数据调整形状，以适应模型输入要求
  const trainXs = d.xs.reshape([1000, 28, 28, 1])
  // 获取训练样本的标签
  const trainYs = d.labels
  // 将测试样本的输入数据调整形状，以适应模型输入要求
  const testXs = t.xs.reshape([200, 28, 28, 1])
  // 获取测试样本的标签
  const testYs = t.labels
  // 返回格式化后的训练和测试数据
  return {
    trainXs,
    trainYs,
    testXs,
    testYs
  }
}

/**
 * 异步函数，用于展示MNIST数据集中的图像数据
 * 此函数首先创建一个MnistData实例，然后加载数据集
 * 接着，它获取一个包含20个训练样本的批次，并将这些样本渲染到可视化界面中
 */
const showData = async () => {
  // 创建MnistData实例
  const data = new MnistData()
  // 加载MNIST数据集
  await data.load()
  // 获取20个训练样本
  const e = data.nextTrainBatch(20)
  // 创建可视化界面的表面，用于展示图像数据
  const surface = tfvis.visor().surface({
    name: '输入数据',
  })
  // 遍历20个样本，将每个样本渲染为图像
  for (let i = 0; i < 20; i++) {
    // 使用tf.tidy清理临时张量，防止内存泄漏
    const imgTensor = tf.tidy(() => {
      // 从批次中提取单个样本的图像数据，并将其重塑为28x28的图像张量
      return e.xs.slice([i, 0], [1, 784]).reshape(([28, 28, 1]))
    })
    // 创建canvas元素，用于渲染图像
    const canvas = document.createElement('canvas')
    canvas.width = 28
    canvas.height = 28
    canvas.style = 'margin: 4px'
    // 将图像张量渲染到canvas上
    await tf.browser.toPixels(imgTensor, canvas)
    // 将渲染后的canvas添加到可视化界面的表面
    surface.drawArea.appendChild(canvas)
  }
}


/**
 * 异步函数用于训练模型
 *
 * 该函数接受一个模型以及训练和测试用的输入和输出数据，编译并训练模型
 * 训练过程中，将使用TensorFlow.js的可视化工具来展示训练和验证过程的损失和准确率
 *
 * @param {tf.LayersModel} model - 需要训练的TensorFlow.js模型
 * @param {tf.Tensor} trainXs - 训练用的输入数据
 * @param {tf.Tensor} trainYs - 训练用的输出数据
 * @param {tf.Tensor} testXs - 测试用的输入数据
 * @param {tf.Tensor} testYs - 测试用的输出数据
 * @returns {Promise<tf.History>} - 返回训练历史记录的Promise对象，包含训练和验证过程的损失和准确率
 */
const trainModel = async (model, trainXs, trainYs, testXs, testYs,) => {
  // 编译模型，设置优化器、损失函数和评估指标
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  // 训练模型，并返回训练历史记录的Promise对象
  // 训练过程中使用提供的测试数据进行验证，设置训练轮次和回调函数
  return await model.fit(trainXs, trainYs, {
    validationData: [testXs, testYs],
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks(
        {name: '训练过程'},
        ['loss', 'val_loss', 'acc', 'val_acc'],
      {
        callbacks: ['onEpochEnd'],
      }
    ),
  })
}


/**
 * 根据给定的模型测试用户手写数字识别
 * @param {Object} model - 用于数字识别的预训练模型
 */
const testModel = (model)=>{
  // 获取画布元素和2D渲染上下文
  const canvas = document.getElementById('canvas')
  const ctx = canvas.getContext('2d')
  // 初始化画布背景为黑色
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  // 监听鼠标移动事件以绘制白色方块
  canvas.addEventListener('mousemove',(e)=>{
    // 当鼠标左键被按下时绘制
    if (e.buttons === 1){
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = 'white'
      ctx.fillRect(e.offsetX, e.offsetY, 25, 25)
    }
  })

  // 清除画布的方法
  window.clear = ()=>{
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
  // 使用模型预测画布上的数字
  window.predict = (form)=>{
    // 将画布内容转换为模型输入格式
    const inputs = tf.tidy(()=>{
      return tf.image.resizeBilinear(
        tf.browser.fromPixels(canvas),
        [28, 28],
        true
      ).slice([0, 0, 0], [28, 28, 1]).toFloat().div(255).reshape([1, 28, 28, 1])

    })
    // 使用模型进行预测并显示结果
    const pred = model.predict(inputs).argMax(1)
    alert(pred.dataSync()[0])
  }
}

/**
 * 执行MNIST数据集的加载、处理、模型训练和测试的异步函数
 * 此函数作为程序的主流程，确保数据处理和模型训练按顺序执行
 */
const runMnist = async () => {
  // 加载并展示MNIST数据集信息
  await showData()

  // 加工数据，将其转换为适合模型训练的格式
  const {trainXs, trainYs, testXs, testYs} = await covertData()

  // 创建并配置神经网络模型
  const model = createModel()

  // 训练模型，并使用测试数据集评估模型性能
  await trainModel(model, trainXs, trainYs, testXs, testYs)

  // 使用训练好的模型进行预测测试
  testModel(model)
}


window.onload = async () => {
  await runMnist()
}
