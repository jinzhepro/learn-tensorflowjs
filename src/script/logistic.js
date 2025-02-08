import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import generateBinaryClassificationDataset from "./data";

async function runLogisticRegression() {
  const data = generateBinaryClassificationDataset(400)
  await tfvis.render.scatterplot({
    name: '逻辑回归训练数据'
  },{
    values: [
      data.filter(d => d.label === 0),
      data.filter(d => d.label === 1)
    ],
    series: ['数据1', '数据2']
  })
}

window.onload = ()=>{
  runLogisticRegression().then(() => console.log('done'))
}
