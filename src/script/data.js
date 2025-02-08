function generateBinaryClassificationDataset(numSamples) {
  let dataset = [];

  for (let i = 0; i < numSamples; i++) {
    // 随机生成两个特征值
    let feature1 = Math.random() * 20 - 10; // 特征1在-10到10之间
    let feature2 = Math.random() * 20 - 10; // 特征2在-10到10之间

    // 假设有一个简单的线性决策边界: feature2 > feature1
    // 根据该条件判断样本属于哪一类
    let label = feature2 > feature1 ? 1 : 0;

    // 将生成的样本和标签加入到数据集中
    dataset.push({
      x: feature1,
      y: feature2,
      label: label
    });
  }

  return dataset;
}
export default generateBinaryClassificationDataset;
