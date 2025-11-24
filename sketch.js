// slider operations:
// translates in local space
// rotates around origin
// scales from origin


// idea. Maybe the sliders can be moved and rotated around to allow the player to create their own workstation that's 
// more intuitive for them. This would allow them to take ownership of the work.
// in addition to moving and rotating them, have a curve value that can create bends as well as a circle.

// after computing the new slider configuration, we can iterate over each slider at a time and interpolate to its new value
// hopefully showing a nonsensical set of transformations that eventually converge to the desired one.

const sliderCount = 6;
let sliders = [];
let matrices = [];
matricesTensors = []; // tf.Tensor2D versions of matrices

let img;
let M; // our 3x3 matrix

let regressor;
const ROWS = 3;
const COLS = 3;
let lastLoss = 0;
let lastPred = [];
let training = false;

function mousePressed() {
  sliders.forEach(slider => slider.handleMousePressed());
}

function mouseDragged() {
  sliders.forEach(slider => slider.handleMouseDragged());
}

function mouseReleased() {
  sliders.forEach(slider => slider.handleMouseReleased());
}

async function setup() {
  createCanvas(1400, 700);

  for (let i = 0; i < sliderCount; i++) {
    //random 
    v = random(-1.0, 1.0);
    
    sliders[i] = new FancySlider(150, 30 + i * 30, 200, i % 2 == 0 ? 0 : PI);
    // contune training when user releases slider

      matrices[i] = createRandomMatrix();
    matricesTensors[i] = mathMatrixToTFMatrix(matrices[i]); // tf.Tensor2D
  
  }

  img = loadImage('sprite.jpg');
  
  noStroke();
  
  
  await tf.ready();

  regressor = new MatrixRegressor({
    inRows: ROWS,
    inCols: COLS,
    outputSize: sliderCount,
    hiddenUnits: 9,
    learningRate: 0.005,
    lossFn: (yPred, context) => {   
      let m = matrixFromSlidersTF(yPred, matricesTensors); // tf.Tensor2D [3,3]
      const target = mathMatrixToTFMatrix(context.inputMatrix);
      const diff = m.sub(target);     // tf.sub
      const absDiff = diff.abs();     // elementwise abs
      const loss = absDiff.sum();     // scalar
      return loss; 
    }
  });

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

function getNormalMatrix()
{
  
  // ----- build an affine transform matrix -----
  // We'll: scale, rotate, then translate.
  const angle = 1.0 * Math.PI; // rotation angle in radians
  const sx = 1.0;
  const sy = 1.0;
  const tx = 1.0 * 100; // translate x
  const ty = 1.0 * 100; // translate y

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

  // alert(matrixToString(randM));

  return randM;
}

function updateMatrix() {
  
  M = math.identity(3);
  for (let i = 0; i < matrices.length; i++)
  {

    var v = sliders[i].value();
    var newMatrix;
    if( i == 0 )
      newMatrix = getTransformMatrix(0, 1+v, 1+v, 0, 0);
      else if( i == 1 )
    newMatrix = getTransformMatrix(v*math.PI, 1, 1, 0, 0);
  else if( i == 2 )
    newMatrix = getTransformMatrix(0, 1, 1, 0, v);
    if( i == 3 )
      newMatrix = getTransformMatrix(0, 1, 1, 0, v);
    else if( i == 4 )
      newMatrix = getTransformMatrix(v*math.PI, 1, 1, 0, 0);
    else if( i == 5 )
      newMatrix = getTransformMatrix(0, 1+v, 1+v, 0, 0);

    M = math.multiply(newMatrix, M);
  }
}

// sliderTensor: tf.Tensor of shape [1, N] or [N]
// matrices: JS array of length N, each element is a 3x3 JS array
function matrixFromSlidersTF(slidersTensor, matricesTensors) {
  const numSliders = matricesTensors.length;

  // Ensure shape [numSliders]
  const sliders = slidersTensor.reshape([numSliders]); // [N]

  // Start from 3x3 zeros tensor
  let M = tf.zeros([3, 3]);

  for (let i = 0; i < numSliders; i++) {
    // scalar weight for this matrix
    const w = sliders.slice([i], [1]).reshape([]); // scalar
    const r = w.sub(0.5).mul(2.0); // remap 0..1 to -1..1 using tensor ops
    const mat = matricesTensors[i];  // tf.Tensor2D [3,3]
    // weighted submatrix
    const subMatrix = mat.mul(r);                  // [3,3]

    // accumulate
    M = M.add(subMatrix);                          // [3,3]
  }

  return M; // tf.Tensor2D [3,3]
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

let trainingLoss;
function draw() 
{
  background(220);


  if (training)
  {
    inputMatrix = getTransformMatrix(random(-Math.PI, Math.PI), random(-2, 2), random(-2, 2), 0, 0);
    inputMatrix = getTransformMatrix(0, 1.1, 1.1, 0, 0);
        
    trainingLoss = trainingStep(inputMatrix);
  }

  updateMatrix();

  // M = getNormalMatrix();

  hw = width / 2;
  hh = height / 2;

  M.set([0,2], (M.get([0,2]) * hw));
  M.set([1,2], (M.get([1,2]) * hw));

  // M = lerpMatrix(M, getNormalMatrix(), sliders[4].value());



  // Extract the 2D affine part for p5's applyMatrix(a, b, c, d, e, f)


  push();
  // p5 uses:
  // x' = a*x + c*y + e
  // y' = b*x + d*y + f


  applyMatrixM(M);

  
  // Draw the image in "local" coordinates (0,0)
  image(img, 0, 0);
  pop();  

  let h = 30 * (sliderCount + 1);

  text("Current Transformation Matrix:\n" + matrixToString(M), 10, h);

  h += 80;
  text("Training loss: " + nf(trainingLoss, 1, 4), 10, h);

  

  sliders.forEach(slider => slider.draw());

}

function trainingStep(inputMatrix) // sample input matrix
{
  // Train one step with this matrix
  lastLoss = regressor.trainStep(inputMatrix);

  // Get prediction for visualization
  lastPred = regressor.predict(inputMatrix);

  for (let i = 0; i < lastPred.length; i++)
    sliders[i].value(lastPred[i] * 2.0 - 1.0); // set slider to match prediction;
  
  return lastLoss;
}

// Simple helpers for the sketch
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