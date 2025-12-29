class CutoutLibrary {
  constructor(jsonPath) {
    this.cutouts = [];
    // Load the JSON file
    loadJSON(jsonPath, (data) => {
      // Iterate over the array in the JSON
      for (let item of data) {
        this.cutouts.push(new Cutout(item));
      }
    });
  }

  // Get a specific cutout by index
  get(index) {
    if (index >= 0 && index < this.cutouts.length) {
      return this.cutouts[index];
    }
    return null;
  }
  
  // Get the number of cutouts
  count() {
    return this.cutouts.length;
  }
}

class Cutout {
  constructor(data) {
    this.imageName = data.imageName;
    this.scaleFactor = data.scale || 1.0;
    
    // Parse the target matrix from the JSON array
    // Assuming data.targetMatrix is a 3x3 array of numbers
    this.targetMatrix = math.matrix(data.targetMatrix);

    // Load the image
    this.img = loadImage(this.imageName);
  }

  // Draws the image as a silhouette at the target matrix.
  // Checks if inputMatrix is close to targetMatrix.
  // Returns true if close, false otherwise.
  drawTarget(inputMatrix) {
    let {isClose, diffSum} = this.isCloseEnough(inputMatrix);

    tint(0, 150);
    this.drawAtTarget();
    tint(255);

    // draw actual image //
  	this.drawAt(inputMatrix);

    return {isClose, diffSum};
  }

  // Draws the image using a given position matrix
  drawAt(positionMatrix) {

    let copyMatrix = positionMatrix.clone();

    push();

    console.assert(width && height, "Canvas size must be defined");

    const cw = width / 2;
    const ch = height / 2;
    
    // center in canvas and scale position matrix to canvas coords
    copyMatrix.set([0,2], (copyMatrix.get([0,2]) * ch + cw));
    copyMatrix.set([1,2], (copyMatrix.get([1,2]) * ch + ch));
    
    // Apply the given position matrix
    this.applyMathMatrix(copyMatrix);
    
    // Apply scale factor
    scale(this.scaleFactor);
    
    // Draw the image centered
    image(this.img, -this.img.width / 2, -this.img.height / 2);
    
    pop();

  }

  drawAtTarget()
  {
	  this.drawAt(this.targetMatrix);
  }

  // Helper to check if input matrix is close to target matrix
  isCloseEnough(inputMatrix) {
    if (!inputMatrix) return false;

    // Calculate element-wise difference
    let diff = math.subtract(this.targetMatrix, inputMatrix);
    
    // Threshold for "close enough"
    // Adjust this value based on sensitivity requirements
    
    let diffSum = 0;

    diff.forEach(function (value) {
      diffSum += Math.abs(value);
    });
    
    return {isClose: diffSum < detectionThreshold, diffSum: diffSum};
  }

  // Helper to apply a math.js matrix to p5 context
  applyMathMatrix(M) {
    // M is expected to be a 3x3 math.js matrix
    // [ a c e ]
    // [ b d f ]
    // [ 0 0 1 ]
    
    const a = M.get([0, 0]);
    const c = M.get([0, 1]);
    const e = M.get([0, 2]);
    const b = M.get([1, 0]);
    const d = M.get([1, 1]);
    const f = M.get([1, 2]);

    applyMatrix(a, b, c, d, e, f);
  }
}
