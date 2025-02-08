import * as tf from '@tensorflow/tfjs'

const t0 = tf.tensor(1)
t0.print()
console.log(t0)

const t1 = tf.tensor([1,2])
t1.print()
console.log(t1)

const t2 = tf.tensor([[1,2], [3,4], [5,6]])
t2.print()
console.log(t2)

const t3 = tf.tensor([[[1,3,5], [2,4,6]]])
t3.print()
console.log(t3)

const t4 = tf.tensor([[[[1,3,5], [2,4,6]]]])
t4.print()
console.log(t4)

const input = [1,2,3,4]
const w = [[1,8,6,9],[2,3,8,0],[8,6,2,7],[1,4,9,5]]
const output = [0,0,0,0]

for (let i = 0; i < w.length; i++){
  for (let j = 0; j < input.length; j++){
    output[i] += w[i][j] * input[j]
  }
}

console.log(output)

tf.tensor(w).dot(tf.tensor(input)).print()
