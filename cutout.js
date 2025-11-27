class CutoutLibrary {
  constructor(jsonPath, canvasSize) {
    this.cutouts = [];
    this.canvasSize = canvasSize;
    // Load the JSON file
    loadJSON(jsonPath, (data) => {
      // Iterate over the array in the JSON
      for (let item of data) {
        this.cutouts.push(new Cutout(item, canvasSize));
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
  constructor(data, canvasSize) {
    this.imageName = data.imageName;
    this.scaleFactor = data.scale || 1.0;;
    
    // Parse the target matrix from the JSON array
    // Assuming data.targetMatrix is a 3x3 array of numbers
    this.targetMatrix = math.matrix(data.targetMatrix);

	let hw = canvasSize.width / 2;
	let hh = canvasSize.height / 2;

	this.targetMatrix.set([0,2], (this.targetMatrix.get([0,2]) + hw));
	this.targetMatrix.set([1,2], (this.targetMatrix.get([1,2]) + hh));
	
    // Load the image
    this.img = loadImage(this.imageName);
  }

  // Draws the image as a silhouette at the target matrix.
  // Checks if inputMatrix is close to targetMatrix.
  // Returns true if close, false otherwise.
  drawTarget(inputMatrix) {
    let isClose = this.isCloseEnough(inputMatrix);

    push();
    
    // Apply the target matrix transformation
    this.applyMathMatrix(this.targetMatrix);
    
    // Apply scale factor if needed (assuming it scales the image size)
    scale(this.scaleFactor);

    // Set color based on closeness
    if (isClose) {
      // Green with some transparency
      tint(0, 255, 0, 200);
    } else {
      // Gray with some transparency
      tint(100, 200);
    }

    // Draw the image centered
    image(this.img, -this.img.width / 2, -this.img.height / 2);

    pop();

  // draw actual image //

  if (!isClose) 
    tint(225); 
  else 
    noTint();

  this.drawAt(inputMatrix);

    return isClose;
  }

  // Draws the image using a given position matrix
  drawAt(positionMatrix) {
    push();
    
    // Apply the given position matrix
    this.applyMathMatrix(positionMatrix);
    
    // Apply scale factor
    scale(this.scaleFactor);
    
    // Draw the image centered
    image(this.img, -this.img.width / 2, -this.img.height / 2);
    
    pop();
  }

  // Helper to check if input matrix is close to target matrix
  isCloseEnough(inputMatrix) {
    if (!inputMatrix) return false;

    // Calculate element-wise difference
    let diff = math.subtract(this.targetMatrix, inputMatrix);
    
    // Calculate sum of absolute differences
    let sum = 0;
    diff.forEach(function (value) {
      sum += Math.abs(value);
    });

    // Threshold for "close enough"
    // Adjust this value based on sensitivity requirements
    const threshold = 50.0; 
    
    return sum < threshold;
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
