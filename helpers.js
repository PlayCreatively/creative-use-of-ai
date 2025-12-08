const { get } = require("http");

function applyMatrixM(M)
{
  const a = M.get([0, 0]);
  const c = M.get([0, 1]);
  const e = M.get([0, 2]);
  const b = M.get([1, 0]);
  const d = M.get([1, 1]);
  const f = M.get([1, 2]);

  applyMatrix(a, b, c, d, e, f);
}

function lerpMatrix(M1, M2, t)
{
  let result = math.zeros(3,3);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      result.set([i,j], M1.get([i,j]) + (M2.get([i,j]) - M1.get([i,j])) * t);
    }
  }
  return result;
}


function matrixToString(M) {
  let result = "";
  for (let r = 0; r < 3; r++) 
  {
    let rowStr = "";
    for (let c = 0; c < 3; c++) 
      rowStr += M.get([r, c]).toFixed(2) + "\t";

    result += rowStr + "\n";
  }
  return result;
}


function createRandomMatrix() {
  const min = 0.5;
  const max = 1.0;

  let bitflag = Math.floor(random(0, 63));

  // alert(bitflag);

  let randM = math.zeros(3,3);

  for (let i = 0; i < 6; i++)
    if ((bitflag & (1 << i)) == 0)
    {
      let randVal = random(min, max) * (random() < 0.5 ? -1 : 1);
      randM.set([Math.floor(i / 3), i % 3], randVal);
    }

  return randM;
}


function getTransformMatrix(angle, sx, sy, tx, ty)
{
  
  // ----- build an affine transform matrix -----
  // We'll: scale, rotate, then translate.

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  // Standard 2D affine form:
  // [ a c e ]
  // [ b d f ]
  // [ 0 0 1 ]
  const a =  cosA * sx;
  const b =  sinA * sx;
  const c = -sinA * sy;
  const d =  cosA * sy;
  const e = tx;
  const f = ty;

  return math.matrix([
    [a, c, e],
    [b, d, f],
    [0, 0, 1]
  ]);
}

function getTranslationMatrix(tx, ty)
{
  return math.matrix([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
  ]);
}

function getScaleMatrix(sx, sy)
{
  return math.matrix([
    [sx, 0,  0],
    [0,  sy, 0],
    [0,  0,  1]
  ]);
}

function getRotationMatrix(angle)
{
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  return math.matrix([
    [ cosA, -sinA, 0],
    [ sinA,  cosA, 0],
    [ 0,     0,    1]
  ]);
}

function mathMatrixToTFMatrix(M)
{
  const [rows, cols] = M.size();
  const values = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      values.push(M.get([r,c]));

  return tf.tensor2d(values, [rows, cols]);
}

function randomMatrix(rows, cols) {
  const m = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      row.push(random()); // p5 random 0..1
    }
    m.push(row);
  }
  return m;
}

function matrixMean(mat) {
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

function matrixAbsSum(mat) {
  let sum = 0;
  for (let r = 0; r < mat.length; r++)
    for (let c = 0; c < mat[r].length; c++)
      sum += Math.abs(mat[r][c]);
  return sum;
}

// sliderTensor: tf.Tensor of shape [1, N] or [N]
function matrixFromSliders(sliderValues) {
  return tf.tidy(() => {
    // Ensure input is 2D: [batchSize, 6]
    let vals = sliderValues;
    if (sliderValues.rank === 1) {
      vals = sliderValues.expandDims(0);
    }
    
    const batchSize = vals.shape[0];
    const zeros = tf.zeros([batchSize, 1]);
    const ones = tf.ones([batchSize, 1]);

    // Split sliders into individual columns [batch, 1]
    const [s0, s1, s2, s3, s4, s5] = tf.split(vals, 6, 1);

    // Helper: Build 3x3 matrix from components
    // [ sx*cos -sy*sin tx ]
    // [ sx*sin  sy*cos ty ]
    // [ 0       0      1  ]
    const makeMat = (angle, sx, sy, tx, ty) => {
       const c = tf.cos(angle);
       const s = tf.sin(angle);
       
       // Stack all 9 elements at once, then reshape
       return tf.stack([
         c.mul(sx), s.mul(sy).neg(), tx,
         s.mul(sx), c.mul(sy),       ty,
         zeros,     zeros,           ones
       ], 1).reshape([batchSize, 3, 3]);
    };

    // --- 1. Scale (1+v, 1+v) ---
    let M = makeMat(zeros, s0.add(1), s0.add(1), zeros, zeros);

    // --- 2. Translate Y (v) ---
    let M1 = makeMat(zeros, ones, ones, zeros, s1);
    M = M1.matMul(M);

    // --- 3. Rotate (v*PI) ---
    let M2 = makeMat(s2.mul(Math.PI), ones, ones, zeros, zeros);
    M = M2.matMul(M);

    // --- 4. Scale (1+v, 1+v) ---
    let M3 = makeMat(zeros, s3.add(1), s3.add(1), zeros, zeros);
    M = M3.matMul(M);

    // --- 5. Special Rotation (HalfAngle1 + HalfAngle2) ---
    const PI = Math.PI;
    
    // HalfAngle1 (s4) -> [cos, sin, 0; 0,0,0; 0,0,1]
    const a4 = s4.mul(PI);
    const H1 = tf.stack([
        tf.cos(a4), tf.sin(a4), zeros,
        zeros,      zeros,      zeros,
        zeros,      zeros,      ones
    ], 1).reshape([batchSize, 3, 3]);

    // HalfAngle2 (s5) -> [0,0,0; -sin, cos, 0; 0,0,0]
    const a5 = s5.mul(PI);
    const H2 = tf.stack([
        zeros,             zeros,      zeros,
        tf.sin(a5).neg(),  tf.cos(a5), zeros,
        zeros,             zeros,      zeros
    ], 1).reshape([batchSize, 3, 3]);

    // Combine and multiply
    M = M.matMul(H1.add(H2));

    return M;
  });
}

function stringContainsAny(mainStr, subStrings) {
  return subStrings.some(subStr => mainStr.indexOf(subStr) !== -1);
}