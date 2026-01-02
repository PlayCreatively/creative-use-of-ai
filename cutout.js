class CutoutLibrary {
  constructor(jsonPath) {
    this.cutouts = [];
    this.background = [];
    // Load the JSON file
    loadJSON(jsonPath, (data) => {
      // Iterate over the array in the JSON
      let order = 0;
      for (let cutout of data.background) {
        if(cutout.order === undefined) {
          cutout.order = order;
          order++;
        }
        this.background.push(new Cutout(cutout));
      }
      for (let cutout of data.cutouts) {
        if(cutout.order === undefined) {
          cutout.order = order;
          order++;
        }
        this.cutouts.push(new Cutout(cutout));
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

  drawAllBefore(index, drawFunc)
  {
    let drawList = this.cutouts.slice(0, index)
      .concat(this.background);
    
    const indexCutoutOrder = this.cutouts[index].order;
    let funcDrawn = false;
    
    // Sort by order
    drawList.sort((a, b) => a.order - b.order);

    for(let cutout of drawList)
    {
      if(!funcDrawn && indexCutoutOrder <= cutout.order)
      {
        drawFunc(this.cutouts[index]);
        funcDrawn = true;
      }

      tint(150);
      cutout.drawAtTarget();
      tint(255);
    }

    if(!funcDrawn)
      drawFunc(this.cutouts[index]);
  }
}

class Cutout {
  constructor(data) {
    this.imageName = data.imageName;
    this.scaleFactor = data.scale || 1.0;
    this.order = data.order;
    
    // Parse the target matrix from the JSON array
    // Assuming data.targetMatrix is a 3x3 array of numbers
    this.targetMatrix = math.matrix(data.targetMatrix);

    // Load the image
    this.img = loadImage("../images/" + this.imageName, (loadedImg) => {
      this.silhouetteImg = loadedImg.get();
      this.silhouetteImg.loadPixels();
      for (let i = 0; i < this.silhouetteImg.pixels.length; i += 4) {
        this.silhouetteImg.pixels[i] = 255;     // R
        this.silhouetteImg.pixels[i + 1] = 255; // G
        this.silhouetteImg.pixels[i + 2] = 255; // B
        // Alpha (i+3) remains unchanged
      }
      this.silhouetteImg.updatePixels();
    });
  }

  drawOutline(matrix, weight = 2) {
    const steps = 8;
    for (let i = 0; i < steps; i++) {
      const angle = (TWO_PI * i) / steps;
      const x = cos(angle) * weight;
      const y = sin(angle) * weight;
      this.drawAt(matrix, x, y, true);
    }
  }

  // Draws the image as a silhouette at the target matrix.
  // Checks if inputMatrix is close to targetMatrix.
  // Returns true if close, false otherwise.
  drawTarget(inputMatrix, r = 0, g = 0, b = 0) {
    tint(r, g, b, 255);
    this.drawOutline(this.targetMatrix, 2);
    tint(255);

    // draw actual image //
  	this.drawAt(inputMatrix);

    return this.getDiffSum(inputMatrix);
  }

  // Draws the image using a given position matrix
  drawAt(positionMatrix, x = 0, y = 0, useSilhouette = false) {

    push();

    console.assert(width && height, "Canvas size must be defined");

    const cw = width / 2;
    const ch = height / 2;
    
    translate(cw + positionMatrix.get([0,2]) * ch, ch + positionMatrix.get([1,2]) * ch);
    
    // Apply rotation/scale/shear
    const a = positionMatrix.get([0, 0]);
    const c = positionMatrix.get([0, 1]);
    const b = positionMatrix.get([1, 0]);
    const d = positionMatrix.get([1, 1]);
    applyMatrix(a, b, c, d, 0, 0);
    
    // Apply scale factor
    scale(this.scaleFactor);
    
    // Draw the image centered
    imageMode(CENTER);
    
    if (useSilhouette && this.silhouetteImg) {
      image(this.silhouetteImg, x, y);
    } else {
      image(this.img, x, y);
    }
    
    pop();

  }

  drawAtTarget()
  {
	  this.drawAt(this.targetMatrix);
  }

  // Helper to check if input matrix is close to target matrix
  getDiffSum(inputMatrix) {
    // Calculate element-wise difference
    let diff = math.subtract(this.targetMatrix, inputMatrix);
    
    // Threshold for "close enough"
    // Adjust this value based on sensitivity requirements
    
    let diffSum = 0;

    diff.forEach(function (value) {
      diffSum += Math.abs(value);
    });
    
    return diffSum;
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
