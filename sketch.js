const sliderCount = 5;
let sliders = [];
let selectedSlider = null;

let M; // our 3x3 matrix

let width = 1400;
let height = 700;

const windowed = {width: 1400, height: 700};

const detectionThreshold = .05;

let regressor;
const ROWS = 3;
const COLS = 3;
let lastLoss = 0;
let lastPred = [];
let aiCoroutine = null;

let aiTargetMatrix = math.identity(3);

let currentCutoutIndex = 0;
let library;
const confirmButton = new SimpleButton();
const fullscreenButton = new SimpleButton();
let chatLine;
let isTraining = false;

function preload() {
  library = new CutoutLibrary("cutouts.json");
}

function windowResized() 
{  
  // Check if p5 thinks we are fullscreen OR if the window size matches the screen size (F11)
  const isF11 = windowWidth >= screen.width && windowHeight >= screen.height;

  const fs = fullscreen() || isF11;
  width = fs ? windowWidth : windowed.width;
  height = fs ? windowHeight : windowed.height;
  resizeCanvas(width, height);
}

function mousePressed() {
  const lastSelectedSlider = selectedSlider;
  selectedSlider = null;

  sliders.forEach(slider => { if (slider.handleMousePressed(slider === lastSelectedSlider)) selectedSlider = slider; });
  
  if (chatLine) chatLine.mousePressed();

  if (fullscreenButton.isHovering(mouseX, mouseY)) {
    const fs = !fullscreen();
    fullscreen(fs);
  }

  if (confirmButton.isHovering(mouseX, mouseY)) {

    // Save current matrix as target for current cutout
    library.get(currentCutoutIndex).targetMatrix = M.clone();

    currentCutoutIndex++;
    resetSliders();
    if (currentCutoutIndex >= library.count()) {
        currentCutoutIndex = 0; 
    }
  }
}

function mouseDragged() {
  sliders.forEach(slider => { if (slider.handleMouseDragged(slider === selectedSlider)) selectedSlider = slider; });
}

function mouseReleased() {
  sliders.forEach(slider => slider.handleMouseReleased());
}

function resetSliders() {
  sliders.forEach(slider => slider.value(random(-1.0, 1.0)));
}

async function setup() {
  createCanvas(width, height);

  for (let i = 0; i < sliderCount; i++) {
    //random 
    v = random(-1.0, 1.0);
    
    sliders[i] = new FancySlider(150, 30 + i * 30, 200, i % 2 == 0 ? 0 : PI);
  }

  // resetSliders();
  
  // Initialize ChatLine at the bottom
  chatLine = new ChatLine(400, 30, handleChatCommand);
  
  noStroke();
  
  await tf.ready();

  regressor = new MatrixRegressor({
    inRows: ROWS,
    inCols: COLS,
    outputSize: sliderCount,
    hiddenUnits: 1,
    learningRate: 0.05,
    lossFn: (yPred, context) => {   
      // Map 0..1 output to -1..1 range to match sliders
      const mappedPred = yPred.mul(2).sub(1);

      let m = calculateModelMatrixTF(mappedPred); 
      const target = mathMatrixToTFMatrix(context.inputMatrix);
      const loss = m.sub(target).abs().sum();
      return loss; 
    }
  });

}



function handleChatCommand(command) {

  if (isTraining)
  {
    isTraining = false; // stop training
    return;
  }

  aiTargetMatrix = parseCommand(command, M);

  if (!aiTargetMatrix)
    return;
  
  aiCoroutine = transformRoutine();
}

function* transformRoutine() {
  isTraining = true;

  regressor.setRestingOutputFromSliders(sliders.map(s => s.value()));
  
  const minLoss = 0.05;
  const maxIterations = 300;
  const movingAvgWindow = 30;
  let iterations = 0;
  let lastLoss = 0;
  let loss = 0;
  let SMA_lossDelta_sum = 0;
  let lossDeltas = [];

  while (isTraining && (loss = trainingStep(aiTargetMatrix)) > minLoss && iterations < maxIterations) {

    const lossDelta = lastLoss - loss;

    // Simple Moving Average over last 10 loss deltas
    SMA_lossDelta_sum += lossDelta;
    lossDeltas.push(lossDelta);
    if (lossDeltas.length > movingAvgWindow) {
      SMA_lossDelta_sum -= lossDeltas.shift();
    }

    const SMA_lossDelta = SMA_lossDelta_sum / movingAvgWindow;

    // JUST FOR DEBUGGING
    // library.get(currentCutoutIndex).drawTarget(aiTargetMatrix);

    iterations++;

    lastLoss = loss;
    yield;
  }

  isTraining = false;
}

function updateMatrix() {
  
  // 1. Calculate matrix using the unified TF logic
  // tf.tidy automatically cleans up the tensors created inside
  const mFlat = tf.tidy(() => {
      const vals = sliders.map(s => s.value());
      const sliderTensor = tf.tensor2d([vals]); // Shape [1, 6]
      const mTensor = calculateModelMatrixTF(sliderTensor);
      return mTensor.dataSync(); // Download to CPU array
  });

  // 2. Reconstruct math.js matrix for drawing
  M = math.matrix([
    [mFlat[0], mFlat[1], mFlat[2]],
    [mFlat[3], mFlat[4], mFlat[5]],
    [mFlat[6], mFlat[7], mFlat[8]]
  ]);
}

let trainingLoss;
function draw() 
{
  background(120);

  // font size
  textSize(32);
  fill(180,50,50);

  text("Do anything with AI!", 650, 30);

  // reset font size
  textSize(13);
  fill(0);


  updateMatrix();

  if (aiCoroutine) {
    if (aiCoroutine.next().done) 
      aiCoroutine = null;
  } 
  else if (M) aiTargetMatrix = M.clone();


  for(let i = 0; i < currentCutoutIndex; i++)
  {
    let cutout = library.get(i);
    if (cutout)
      cutout.drawAtTarget();
  }

  const cutout = library.get(currentCutoutIndex);
  
  if (cutout) {
    
    // Draw the target silhouette and check against current matrix M
    let {isClose, diffSum} = cutout.drawTarget(M);
    
    // Calculate button position
    const s = cutout.scaleFactor;
    const localX = (cutout.img.width / 2) * s * 0.75;
    const localY = (cutout.img.height / 2) * s * 0.75;

    let MClone = math.clone(M);

    const canvasSize = {width: width, height: height};
    const cw = canvasSize.width / 2;
    const ch = canvasSize.height / 2;

    // center in canvas and scale position matrix to canvas height coords
    MClone.set([0,2], (MClone.get([0,2]) * ch + cw));
    MClone.set([1,2], (MClone.get([1,2]) * ch + ch));

    const pos = transformPoint(MClone, localX, localY);

    // Draw closeness percentage
    const maxDiff = 2;
    diffSum = math.max(0, diffSum - (detectionThreshold * maxDiff)); // allow some leeway
    const perc = (Math.max(0, 1 - diffSum / maxDiff)) * 100;
    const isPerfect = perc >= 99.5;

    push(); // push settings

    fill(isPerfect ? 255 : 0);

    // Draw confirm button
    if(isClose)
      confirmButton.draw("Confirm", pos.x + 50, pos.y - 22, 100, 40, isPerfect ? 'white' : 'black');

    textSize(20);
    textStyle(BOLD);
    textAlign(RIGHT, CENTER);
    text(perc.toFixed(0)+"%", pos.x + 40, pos.y);

    pop(); // pop settings

    if (chatLine) chatLine.draw(50, height - 50, isTraining ? "Stop" : "Send");
  }

  sliders.forEach(slider => slider.draw());
  if(selectedSlider)
    selectedSlider.drawGizmos();

  if(!fullscreen())
    fullscreenButton.draw("Fullscreen", width - 120, height - 50, 100, 30);

}

function trainingStep(inputMatrix) // sample input matrix
{
  if (!regressor) return 0;

  // Train one step with this matrix
  lastLoss = regressor.trainStep(inputMatrix);

  // regressor.learningRate *= 0.999; // decay learning rate over time

  text("Target Matrix:\n" + matrixToString(inputMatrix), 500, 50);

  text("loss: " + nf(lastLoss, 1, 4), 500, 250);

  // Get prediction for visualization
  lastPred = regressor.predict(inputMatrix);

  for (let i = 0; i < lastPred.length; i++)
    sliders[i].value(lastPred[i] * 2.0 - 1.0); // set slider to match prediction;
  
  return lastLoss;
}

function calculateModelMatrixTF(sliderValues) {
  return tf.tidy(() => {
    // Ensure sliderValues is 2D tensor [batchSize, sliderCount]
    let vals = sliderValues;
    if (sliderValues.rank === 1) {
      vals = sliderValues.expandDims(0);
    }
    
    const batchSize = vals.shape[0];
    const zeros = tf.zeros([batchSize, 1]);
    const ones = tf.ones([batchSize, 1]);
    const PI = Math.PI;


    // Split sliders into individual columns [batch, 1]
    const [s0, s1, s2, s3, s4] = tf.split(vals, sliderCount, 1);

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
    let M = makeMat(zeros, ones, ones, zeros, zeros);
    const M1 = makeMat(zeros, s0.add(1), s0.add(1), zeros, zeros);
    M = M1.matMul(M);

    // --- 5. Special Rotation (HalfAngle1 + HalfAngle2) ---
    
    const a1 = s2.mul(PI);
    const a2 = s3.mul(PI);
    const M4 = tf.stack([
        tf.cos(a1),       tf.sin(a1), zeros,
        tf.sin(a2).neg(), tf.cos(a2), zeros,
        zeros,            zeros,      ones
    ], 1).reshape([batchSize, 3, 3]);

    M = M4.matMul(M);


    // --- 6. Orbit-like Transformation ---
    
    const a3 = s4.mul(PI);
    const M5 = tf.stack([
        ones,  zeros, tf.cos(a3).mul(s1),
        zeros, ones,  tf.sin(a3).mul(s1),
        zeros, zeros, ones
    ], 1).reshape([batchSize, 3, 3]);

    
    M = M.matMul(M5);

    return M;
  });
}