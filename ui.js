class SimpleButton {
  constructor(backgroundColor = [255, 160], circular = false) {
    this.bounds = null;
    this.circular = circular;
    this.backgroundColor = backgroundColor;
  }

  // Call this inside draw()
  draw(label, x, y, w, h, labelColor = 'black') {
    this.bounds = { x, y, w, h };
    
    push();
    translate(x, y);

    const isHovering = this.isHovering(mouseX, mouseY);

    if (isHovering)
      fill(this.backgroundColor.map(c => c + 100));
    else
      fill(...this.backgroundColor);

    cursor(isHovering ? HAND : ARROW)
    
    if(this.circular)
      ellipse(w/2, h/2, w, h);
    else
      rect(0, 0, w, h, 8);
    
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
