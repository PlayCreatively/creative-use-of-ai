class SimpleButton {
  constructor() {
    this.bounds = null;
  }

  // Call this inside draw()
  draw(label, x, y, w, h, labelColor = 'black') {
    this.bounds = { x, y, w, h };
    
    push();
    translate(x, y);

	if(this.isHovering(mouseX, mouseY))
	  fill(250);
    else
      fill(200);
    
    rect(0, 0, w, h, 5);
    
    fill(0);
    textAlign(CENTER, CENTER);
    textSize(16);
    fill(labelColor);
    text(label, w/2, h/2);
    
    pop();
  }

  // Call this inside mousePressed()
  isHovering(mx, my) {
    if (!this.bounds) return false;
    return (mx >= this.bounds.x && mx <= this.bounds.x + this.bounds.w &&
            my >= this.bounds.y && my <= this.bounds.y + this.bounds.h);
  }
}

function transformPoint(M, x, y) {
  // M is 3x3 math.matrix
  // [ a c e ]
  // [ b d f ]
  // [ 0 0 1 ]
  
  const a = M.get([0, 0]);
  const c = M.get([0, 1]);
  const e = M.get([0, 2]);
  const b = M.get([1, 0]);
  const d = M.get([1, 1]);
  const f = M.get([1, 2]);
  
  return {
      x: a * x + c * y + e,
      y: b * x + d * y + f
  };
}
