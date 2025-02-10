import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {getIrisData, IRIS_CLASSES} from "./data";
import {model} from "@tensorflow/tfjs";


/**
 * 创建一个神经网络模型
 *
 * 本函数使用TensorFlow.js库构建了一个简单的神经网络模型该模型是一个顺序模型，
 * 包含两个全连接层（Dense layers）第一层有30个神经元，输入形状为4，使用sigmoid激活函数
 * 第二层有3个神经元，使用softmax激活函数，使得输出能够表示不同类别的概率分布
 *
 * @returns {tf.Sequential} 返回一个未编译的神经网络模型
 */
const createModel = ()=>{
  // 创建一个顺序模型
  const model = tf.sequential()

  // 添加第一个全连接层，包含30个神经元，输入形状为4，使用sigmoid激活函数
  model.add(tf.layers.dense({units: 30, inputShape: [4], activation: 'sigmoid'}))

  // 添加第二个全连接层，包含3个神经元，使用softmax激活函数
  model.add((tf.layers.dense({units: 3, activation: 'softmax'})))

  // 返回构建好的模型
  return model
}

/**
 * 定义一个异步函数用于训练模型
 * 该函数接受训练数据、测试数据和一个未编译的模型作为输入，编译模型并进行训练
 * @param {Tensor} trainXs 训练数据的特征输入
 * @param {Tensor} trainYs 训练数据的标签输出
 * @param {Tensor} testXs 测试数据的特征输入，用于验证模型性能
 * @param {Tensor} testYs 测试数据的标签输出，用于验证模型性能
 * @param {tf.LayersModel} model 一个未编译的TensorFlow.js模型
 * @returns {Promise<tf.History>} 返回训练历史记录，包含损失和准确率等信息
 */
const trainsModel = async (trainXs, trainYs, testXs, testYs, model)=>{
  // 编译模型，指定损失函数、优化器和评估指标
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.1),
    metrics: ['accuracy']
  })

  // 训练模型，指定训练数据、测试数据、训练轮数以及回调函数
  // 回调函数用于在训练过程中可视化损失和准确率
  return await model.fit(trainXs, trainYs, {
    epochs: 100,
    validationData: [testXs, testYs],
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss', 'val_loss','acc', 'val_acc'], {
      callbacks: ['onEpochEnd']
    })
  })
}


/**
 * 为给定的模型创建一个测试函数
 * 这个函数将模型作为参数，以便于在不同的上下文中重用
 * 它在window对象上定义了一个新的predict函数，这样就可以在全局范围内访问它
 *
 * @param {Object} model - 一个机器学习模型，用于进行预测
 */
const testModel = (model)=>{
  /**
   * 全局预测函数
   * 这个函数通过收集HTML表单中用户输入的特征值，使用给定的模型进行预测
   * 它首先将表单输入转换为Tensor，然后使用模型预测，并显示预测结果
   *
   * @param {Object} form - 包含用户输入的表单对象
   */
  window.predict = (form)=>{
    // 将表单输入值转换为Tensor，用于模型预测
    const inputs = tf.tensor([
      [form.a.value * 1, form.b.value * 1, form.c.value * 1, form.d.value * 1]
    ])
    // 使用模型进行预测
    const pred = model.predict(inputs)
    // 显示预测结果
    alert(IRIS_CLASSES[pred.argMax(1).dataSync()[0]])
  }
}

/**
 * 异步函数，用于运行iris数据集上的回归分析
 * 该函数包括数据准备、模型创建、模型训练和模型测试四个主要步骤
 */
const runIrisRegression = async () => {
  // 准备数据集，将iris数据按照70%训练集和30%测试集的比例进行划分
  const [trainXs, trainYs, testXs, testYs] = getIrisData(0.30)

  // 创建用于回归分析的模型
  const model = createModel()

  // 训练和测试模型，这里使用了await关键字，因为模型训练可能是一个异步过程
  await trainsModel(trainXs, trainYs, testXs, testYs, model)

  // 对模型进行最终的测试
  testModel(model)
}


window.onload = ()=>{
  runIrisRegression().then(() => console.log('done'))
}
