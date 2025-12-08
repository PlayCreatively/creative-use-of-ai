// MatrixRegressor
// Small tf.js model: matrix -> 6 normalized outputs, custom loss, custom training loop.

class MatrixRegressor {
  /**
   * options:
   *  - inRows, inCols: matrix size
   *  - outputSize: number of outputs (default 6)
   *  - hiddenUnits: hidden layer width
   *  - learningRate: optimizer LR
   *  - lossFn(yPred, context): returns scalar tf.Tensor
   */
  constructor(options = {}) {
    this.inRows = options.inRows ?? 4;
    this.inCols = options.inCols ?? 4;
    this.inputSize = this.inRows * this.inCols;
    this.outputSize = options.outputSize ?? 6;
    this.hiddenUnits = options.hiddenUnits ?? 16;
    this.learningRate = options.learningRate ?? 0.01;

    // Loss function: yPred [1, outputSize], context = arbitrary object
    this.lossFn = options.lossFn || this.defaultLossFn.bind(this);

    this.optimizer = tf.train.adam(this.learningRate);
    this.buildModel();
  }

  buildModel() {
    this.model = tf.sequential();

    this.model.add(
      tf.layers.dense({
        units: this.hiddenUnits,
        inputShape: [this.inputSize],
        activation: "relu",
      })
    );

    
    this.model.add(
      tf.layers.dense({
        units: this.outputSize,
        activation: "sigmoid", // normalized outputs 0..1
      })
    );
  }

  setRestingOutputFromSliders(sliderValues) {
  if (sliderValues.length !== this.outputSize) {
    console.warn(
      "setRestingOutputFromSliders: expected",
      this.outputSize,
      "values, got",
      sliderValues.length
    );
    return;
  }

  const lastLayer = this.model.layers[this.model.layers.length - 1];
  const [W, b] = lastLayer.getWeights(); // keep current W, just change bias

  // 1) sliders: -1..1 → 0..1
  const desired = sliderValues.map(v => (v + 1) / 2);

  // 2) 0..1 → logits
  const eps = 1e-5;
  const logits = desired.map(v => {
    const clipped = Math.min(1 - eps, Math.max(eps, v));
    return Math.log(clipped / (1 - clipped));
  });

  const newB = tf.tensor1d(logits);

  // Keep W, replace only bias
  lastLayer.setWeights([W, newB]);
}
  
  /**
   * Default loss:
   *  - output[0] should approximate the mean of the input matrix
   *  - other outputs are penalized for being large (L2)
   */
  defaultLossFn(yPred, context) {
    const meanInput = context.meanInput ?? 0.5;

    const meanTarget = tf.scalar(meanInput);
    const y0 = yPred.slice([0, 0], [1, 1]).reshape([]);
    const loss0 = y0.sub(meanTarget).square(); // (y0 - mean)^2

    const rest = yPred.slice([0, 1], [1, this.outputSize - 1]);
    const restLoss = rest.square().mean();

    const weightRest = tf.scalar(0.1); // tweak as you like
    const total = loss0.add(restLoss.mul(weightRest));
    return total;
  }

  /**
   * One training step.
   * @param {number[][]} matrix - inRows x inCols array of numbers
   * @param {object} extraContext - optional extra info for lossFn
   * @returns {number} scalar loss value (JS number)
   */
  trainStep(matrix) {
    const flat = this.flattenMatrix(matrix);
    const context = {
      inputMatrix: matrix
    };

    const lossValue = tf.tidy(() => {
      const x = tf.tensor2d([flat], [1, this.inputSize]);

      const lossTensor = this.optimizer.minimize(() => {
        const yPred = this.model.predict(x); // [1, outputSize]
        return this.lossFn(yPred, context);
      }, true); // true = return the loss tensor

      return lossTensor.dataSync()[0];
    });

    return lossValue;
  }

  /**
   * Predict synchronously (returns JS array).
   * @param {number[][]} matrix - inRows x inCols
   * @returns {number[]} length = outputSize
   */
  predict(matrix) {
    const flat = this.flattenMatrix(matrix);

    return tf.tidy(() => {
      const x = tf.tensor2d([flat], [1, this.inputSize]);
      const yPred = this.model.predict(x); // [1, outputSize]
      return Array.from(yPred.dataSync());
    });
  }

  // --- Helpers ---
  flattenMatrix(mat) {
    const out = [];
    const [rows, cols] = mat.size()
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) 
        out.push(mat.get([r, c]));
    return out;
  }

  matrixMean(mat) {
    let sum = 0;
    let count = 0;
    for (let r = 0; r < mat.length; r++) {
      for (let c = 0; c < mat[r].length; c++) {
        sum += mat[r][c];
        count++;
      }
    }
    return sum / count;
  }
}

// If you want it globally:
window.MatrixRegressor = MatrixRegressor;
