class CutoutLibraries {
  constructor (jsonPath) 
  {
    this.libraries = [];
    loadJSON(jsonPath, (data) => {
      for(const libData of data)
        this.libraries.push(new CutoutLibrary(libData));
    });
  }
  
  // Get a specific library by index
  get(index) {
    if (index >= 0 && index < this.libraries.length) {
      return this.libraries[index];
    }
    return null;
  }
  
  // Get the number of libraries
  count() {
    return this.libraries.length;
  }
}

class CutoutLibrary {
  constructor(libData) {
    this.cutouts = [];
    this.background = [];
    // Iterate over the array
    let order = 0;

    if(libData.background)
      for (let cutout of libData.background) {
        cutout.order = cutout.order ?? order++;
        this.background.push(new Cutout(cutout));
      }
    for (let cutout of libData.cutouts) {
      cutout.order = cutout.order ?? order++;
      this.cutouts.push(new Cutout(cutout));
    }
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

  drawAllBeforeAndTarget(index, inputMatrix)
  {
    let drawList = this.cutouts.slice(0, index);

    if(this.background)
      drawList = this.background.concat(drawList);
      
    
    const targetCutout = this.cutouts[index];
    let funcDrawn = false;
    let loss = 0;
    
    // Sort by order
    drawList.sort((a, b) => a.order - b.order);

    for(let cutout of drawList)
    {
      if(!funcDrawn && targetCutout.order <= cutout.order)
      {
        loss = targetCutout.drawTarget(inputMatrix);
        funcDrawn = true;
      }

      tint(150);
      cutout.drawAtTarget();
      tint(255);
    }

    if(!funcDrawn)
    loss = targetCutout.drawTarget(inputMatrix);

    return loss;
  }

  drawAll()
  {
    let drawList = this.cutouts;

    if(this.background)
      drawList = this.background.concat(drawList);
    
    // Sort by order
    drawList.sort((a, b) => a.order - b.order);

    for(let cutout of drawList)
      cutout.drawAtTarget();
  }
}

class Cutout {
  constructor(data) {
    this.imageName = data.imageName;
    this.scaleFactor = data.scale || 1.0;
    this.order = data.order;
    this.cropout = data.cropout;
    
    // Parse the target matrix from the JSON array
    // Assuming data.targetMatrix is a 3x3 array of numbers
    this.targetMatrix = math.matrix(data.targetMatrix);

    // Load the image
    this.img = loadImage("images/" + this.imageName, (loadedImg) => {
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
  drawTarget(inputMatrix) {
    tint(255, 220, 0, 255);
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
    scale(this.scaleFactor * (height / windowed.height));
    
    // Draw the image centered
    imageMode(CENTER);
    
    const img = useSilhouette && this.silhouetteImg ? this.silhouetteImg : this.img;
    if(this.cropout)
      drawImageTile(img, x, y, this.cropout.c, this.cropout.r, this.cropout.i);
    else
      image(img, x, y);
    
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
