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
let img;
let M; // our 3x3 matrix

const width = 1400;
const height = 700;

let regressor;
const ROWS = 3;
const COLS = 3;
let lastLoss = 0;
let lastPred = [];
let aiCoroutine = null;

let aiTargetMatrix = math.identity(3);

let currentCutoutIndex = 0;
let library;
let confirmButton = new SimpleButton();
let chatLine;

function preload() {
  library = new CutoutLibrary("cutouts.json", {width: width, height: height});
}

function mousePressed() {
  sliders.forEach(slider => slider.handleMousePressed());
  
  if (chatLine) chatLine.mousePressed();

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
  sliders.forEach(slider => slider.handleMouseDragged());
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
    // contune training when user releases slider
  }

  resetSliders();
  
  // Initialize ChatLine at the bottom
  chatLine = new ChatLine(50, height - 50, 400, 30, handleChatCommand);
  
  noStroke();
  
  
  await tf.ready();

  regressor = new MatrixRegressor({
    inRows: ROWS,
    inCols: COLS,
    outputSize: sliderCount,
    hiddenUnits: 9,
    learningRate: 0.05,
    lossFn: (yPred, context) => {   
      // Map 0..1 output to -1..1 range to match sliders
      const mappedPred = yPred.mul(2).sub(1);

      // Use the unified TF function
      let m = calculateModelMatrixTF(mappedPred); 
      const target = mathMatrixToTFMatrix(context.inputMatrix);
      const loss = m.sub(target).abs().sum();
      return loss; 
    }
  });

}



function handleChatCommand(command) {

  command = command.toLowerCase();

  const increase = ["lot", "very", "extremely", "more", "bigger", "further", "much", "far"];
  const decrease = ["little", "slightly", "less", "smaller", "reduce", "down", "bit"];

  const multiplier = stringContainsAny(command, increase) ? 2 : stringContainsAny(command, decrease) ? 0.5 : 1.0;

  const moveCmd = ["move", "translate", "shift"];
  const moveRightCmd = ["right", "east"];
  const moveLeftCmd = ["left", "west"];
  const moveUpCmd = ["up", "north"];
  const moveDownCmd = ["down", "south"];

  const translationX = (stringContainsAny(command, moveCmd) ? 1 : 0) 
  * ((stringContainsAny(command, moveRightCmd) ? 1 : 0) + (stringContainsAny(command, moveLeftCmd) ? -1 : 0)) * multiplier * 0.25;

  const translationY = (stringContainsAny(command, moveCmd) ? 1 : 0) 
  * ((stringContainsAny(command, moveDownCmd) ? 1 : 0) + (stringContainsAny(command, moveUpCmd) ? -1 : 0)) * multiplier * 0.25;
  
  const rotateCmd = ["rotate", "turn", "spin", "twist"];
  const rotateLeftCmd = ["left", "counterclockwise"];
  const rotateRightCmd = ["right", "clockwise"];

  const rotation = (stringContainsAny(command, rotateCmd) ? 1 : 0) 
  * (stringContainsAny(command, rotateLeftCmd) ? -1 : 1) * multiplier * Math.PI * 0.25;

  const scaleUp = ["scale up", "enlarge", "bigger"];
  const scaleDown = ["scale down", "shrink", "smaller"];

  const scale = ((stringContainsAny(command, scaleUp) ? 1 : 0) 
  + (stringContainsAny(command, scaleDown) ? -1 : 0)) * multiplier * 0.5;

  let incrementMatrix = getTransformMatrix(0, scale + 1, scale + 1, translationX, translationY);

  incrementMatrix = math.subtract(incrementMatrix, math.identity(3));

  if(math.deepEqual(incrementMatrix, math.zeros(3,3)))
    return; // no-op

 alert("Applying transformation:\n" +
        "Translation: (" + translationX.toFixed(2) + ", " + translationY.toFixed(2) + ")\n" +
        "Rotation: " + rotation.toFixed(2) + " turns\n" +
        "Scale: " + scale.toFixed(2));


  aiTargetMatrix = math.add(aiTargetMatrix, incrementMatrix);
  aiTargetMatrix = math.multiply(aiTargetMatrix, getTransformMatrix(rotation, 1, 1, 0, 0)); // apply rotation

  aiCoroutine = transformRoutine();
}

function* transformRoutine() {
  regressor.setRestingOutputFromSliders(sliders.map(s => s.value()));
  
  const minLoss = 0.11;
  let iterations = 0;
  const maxIterations = 500;

  while (trainingStep(aiTargetMatrix) > minLoss && iterations < maxIterations) {
    iterations++;
    yield;
  }
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

  let cutout = library.get(currentCutoutIndex);
  
  let h = 30 * (sliderCount + 1);

  text("Current Transformation Matrix:\n" + matrixToString(M), 50, h);
  if (cutout) {
    
    // Draw the target silhouette and check against current matrix M
    let success = cutout.drawTarget(M);
    
    // Calculate button position
    const s = cutout.scaleFactor;
    const localX = (cutout.img.width / 2) * s * 0.75;
    const localY = (cutout.img.height / 2) * s * 0.75;

    let MClone = math.clone(M);

    const canvasSize = {width: width, height: height};
    const cw = canvasSize.width / 2;
    const ch = canvasSize.height / 2;

    // center in canvas and scale position matrix to canvas coords
    MClone.set([0,2], (MClone.get([0,2]) * cw + cw));
    MClone.set([1,2], (MClone.get([1,2]) * ch + ch));

  h += 80;
  text("Training loss: " + nf(trainingLoss, 1, 4), 10, h);

  if (chatLine) chatLine.draw();

  sliders.forEach(slider => slider.draw());

} text("Training loss: " + nf(trainingLoss, 1, 4), 10, h);

  

  sliders.forEach(slider => slider.draw());

}

function trainingStep(inputMatrix) // sample input matrix
{
  if (!regressor) return 0;

  // Train one step with this matrix
  lastLoss = regressor.trainStep(inputMatrix);

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