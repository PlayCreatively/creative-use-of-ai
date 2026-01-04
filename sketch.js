const DEBUG = true;

const sliderCount = 5;
let sliders = [];
let selectedSlider = null;

let M; // our 3x3 matrix

let width = 1400;
let height = 700;

const windowed = {width: 1400, height: 700};

let regressor;
const ROWS = 3;
const COLS = 3;
let lastLoss = 0;
let lastPred = [];
let aiCoroutine = null;

let aiTargetMatrix = math.identity(3);


let currentCutoutIndex = 0;
let curLibraryIndex = 0;
let curLibrary;
let libraries;
const confirmButton = new SimpleButton();
const fullscreenButton = new SimpleButton();
let AI_mascotIMG;
let chatLine;
let settingsGUI;
let isTraining = false;
let amplitude;

function preload() {
  libraries = new CutoutLibraries("cutouts.json");

  AI_mascotIMG = loadImage('images/ai mascot [Gemini Generated].png');

  preloadVoiceLines();
}

function windowResized() 
{  
  // Check if p5 thinks we are fullscreen OR if the window size matches the window size (F11)
  const isF11 = windowWidth >= screen.width && windowHeight >= screen.height;

  const fs = fullscreen() || isF11;
  width = fs ? windowWidth : windowed.width;
  height = fs ? windowHeight : windowed.height;
  resizeCanvas(width, height);
}

function mousePressed(e) {
  // Prevent p5 interaction when clicking on DOM elements (like the settings menu)
  if (e && e.target && e.target.tagName !== 'CANVAS') return;

  const lastSelectedSlider = selectedSlider;
  selectedSlider = null;

  let changed = false;
  sliders.forEach(slider => { 
    if (slider.handleMousePressed(slider === lastSelectedSlider)) {
      selectedSlider = slider;
      changed = true;
    }
  });
  if (changed) FancySlider.saveAll(sliders);
  
  if (chatLine) chatLine.mousePressed();

  if (fullscreenButton.isHovering(mouseX, mouseY)) {
    const fs = !fullscreen();
    fullscreen(fs);
  }

  if (confirmButton.isHovering(mouseX, mouseY)) {

    // Save current matrix as target for current cutout
    curLibrary.get(currentCutoutIndex).targetMatrix = M.clone();

    currentCutoutIndex++;
    resetSliders();
  }
}

function SkipCutout(){
    currentCutoutIndex++;
}

function mouseDragged() {
  let changed = false;
  sliders.forEach(slider => { 
    if (slider.handleMouseDragged(slider === selectedSlider)) {
      selectedSlider = slider;
      changed = true;
    }
  });
  if (changed) FancySlider.saveAll(sliders);
}

function mouseReleased() {
  sliders.forEach(slider => slider.handleMouseReleased());
}

function keyPressed() {


  if(DEBUG){
    if (key === 'n' || key === 'N')
    SkipCutout();
  
    if (key === 'i' || key === 'I')
      alert("Current Matrix:\n" + matrixToString(M));
  }
}

function resetSliders() {
  sliders.forEach(slider => slider.value(random(-1.0, 1.0)));
  FancySlider.saveAll(sliders);
}

async function setup() {
  createCanvas(width, height);

  curLibrary = libraries.get(curLibraryIndex);

  // Initialize audio analyzer with smoothing (0.0 - 0.99)
  amplitude = new p5.Amplitude(0.8);

  for (let i = 0; i < sliderCount; i++) 
    sliders[i] = new FancySlider(150, 100 + i * 30, 200, i % 2 == 0 ? 0 : PI);

  FancySlider.loadAll(sliders);

  // resetSliders();
  
  // Initialize ChatLine at the bottom
  chatLine = new ChatLine(400, 30, handleChatCommand);
  
  // Initialize Settings GUI
  settingsGUI = new SettingsGUI();

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

  if(!DEBUG)
  settingsGUI.openModal('New Update!', (container) => {
    
    createP("Introducing v8.0 🥳🥂🍾</br></br>").parent(container);
    createP("We’ve officially retired the <i>old</i> way of working. You know—thinking, tweaking, deciding.").parent(container);
    createP("Welcome to the future, where effort is optional and outcomes are guaranteed. Simply use natural language to describe the transformations you want, sit back, and let our AI handle everything else. No sliders. No settings. No understanding required.").parent(container);
    createP("Why wrestle with tools when you can issue wishes?<br/>Why learn a process when you can skip straight to results?").parent(container);
    createP("With our latest update, you can now simply type in your desired transformations, and our AI will take care of the rest. No more manual adjustments—just pure creativity at your fingertips!").parent(container);
    createP("v8.0 doesn’t just streamline your workflow—it removes it entirely.<br/>Creativity, fully automated. Confidence, pre-installed.").parent(container);

    let caption = createP("Let us do the work...");
    caption.parent(container);
    caption.style('font-weight', 'bold');
    caption.style('font-size', '1.5em');

    // Image
    let img = createImg('images/we can do it [Gemini Generated].jpeg', 'We can do it');
    img.parent(container);
    img.style('display', 'block');
    img.style('width', '40%');
    img.style('margin', '10px auto');
    img.style('border-radius', '5px');
  })
}



function handleChatCommand(command) {

  const isCommand = command && command.trim().length > 0;
  aiTargetMatrix = null;

  if(isCommand)
    aiTargetMatrix = parseCommand(command, M);

  if (isTraining)
    {
    isTraining = false; // stop training

    if(!aiTargetMatrix)
      return;
  }
  else if(!isCommand)
    return;


  if (!aiTargetMatrix && isCommand)
  {
    playVoiceLine("confused");
    return;
  }

  playVoiceLine("command");
  
  aiCoroutine = transformRoutine();
}

function* transformRoutine() {
  isTraining = true;

  regressor.setRestingOutputFromSliders(sliders.map(s => s.value()));
  
  const minLoss = 0.05;
  const maxIterations = 200;
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
  background(60);

  // font size
  textSize(32);
  fill(180,50,50);

  text("Do anything with AI!", 650, 90);

  // AI mascot
  push();
  imageMode(CENTER);
  
  // Audio reactive scaling
  let level = amplitude.getLevel();
  // Map volume (0 to 0.2) to scale (1.0 to 1.2)
  let scaleFactor = map(level, 0, 0.2, 0, 20, true);

  image(AI_mascotIMG, 150, height - 100 - scaleFactor * .5, 130 - scaleFactor, 150 + scaleFactor);
  pop();

  // reset font size
  textSize(13);
  fill(0);
  
  const finishedAllCutouts = currentCutoutIndex >= curLibrary.count();

  if (finishedAllCutouts) {
    curLibrary.drawAll();

    push();
    textStyle(BOLD);
    
    let button = new SimpleButton();
    button.draw("Upload", width / 2 - 50, height - 150, 100, 40);
    if (button.isHovering(mouseX, mouseY) && mouseIsPressed) {
      curLibraryIndex++;
      if (curLibraryIndex >= libraries.count())
        curLibraryIndex = 0;
      curLibrary = libraries.get(curLibraryIndex);
      currentCutoutIndex = 0;
      resetSliders();
    }

    pop();
    return;
  }
  
  updateMatrix();
  
  if (aiCoroutine) {
    if (aiCoroutine.next().done) 
      aiCoroutine = null;
  } 
  else if (M) aiTargetMatrix = M.clone();
  
  let diffSum = curLibrary.drawAllBeforeAndTarget(currentCutoutIndex, M);
  
  
  // Calculate button position
  const cutout = curLibrary.get(currentCutoutIndex);
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

  const scaleRatio = height / windowed.height;

  MClone.set([0,0], MClone.get([0,0]) * scaleRatio);
  MClone.set([0,1], MClone.get([0,1]) * scaleRatio);
  MClone.set([1,0], MClone.get([1,0]) * scaleRatio);
  MClone.set([1,1], MClone.get([1,1]) * scaleRatio);

  const pos = transformPoint(MClone, localX, localY);

  // Draw closeness percentage
  const maxDiff = 2;
  diffSum = math.max(0, diffSum - .08); // allow some leeway
  const perc = (Math.max(0, 1 - diffSum / maxDiff)) * 100;
  const isPerfect = perc >= 99.5;
  
  // Draw confirm button
  if(isPerfect)
    confirmButton.draw("Confirm", pos.x + 50, pos.y - 22, 100, 40, 'white');

  push(); // push settings

  fill(isPerfect ? 255 : 0);
  textSize(20);
  textStyle(BOLD);
  textAlign(RIGHT, CENTER);
  text(perc.toFixed(0)+"%", pos.x + 40, pos.y);

  pop(); // pop settings

  if (chatLine) chatLine.draw(50, height - 50, isTraining ? "Stop" : "Command");

  sliders.forEach(slider => {
    slider.updateEnableStates(settingsGUI.legacySettings.map(s => s.value));
    slider.draw();
  });
  
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