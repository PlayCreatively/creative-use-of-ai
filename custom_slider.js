class FancySlider {
  constructor(x, y, length, angle = 0) {
    this.x = x;
    this.y = y;
    this.length = length;
    this.thickness = 6;

    this.angle = angle;       
    this.v = 0.0;             // -1..1
    this.enabled = true;      

    this.hover = false;
    
    // Interaction flags
    this.draggingThumb = false;
    this.draggingMove = false;
    this.draggingRotate = false; // New flag for rotation
    
    // Offsets to keep interaction smooth
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;
    this.dragAngleOffset = 0;    // New offset to prevent snapping

    this.gizmoSize = 18;
    this.gizmoMargin = 32;    
  }

  value(newValue) {
    if (newValue === undefined) {
      return this.v;
    }
    this.v = constrain(newValue, -1, 1);
    return this.v;
  }

  // --- coordinate helpers (local slider space vs screen) ---

  screenFromLocal(px, py) {
    let c = cos(this.angle);
    let s = sin(this.angle);
    let sx = this.x + px * c - py * s;
    let sy = this.y + px * s + py * c;
    return createVector(sx, sy);
  }

  localFromScreen(mx, my) {
    let dx = mx - this.x;
    let dy = my - this.y;
    let c = cos(this.angle);
    let s = sin(this.angle);
    let lx = dx * c + dy * s;
    let ly = -dx * s + dy * c;
    return createVector(lx, ly);
  }

  // --- hit tests ---

  isMouseOverTrack() {
    let p = this.localFromScreen(mouseX, mouseY);
    return p.x >= -this.length / 2 &&
           p.x <=  this.length / 2 &&
           (!this.enabled ? (p.y > 0 && p.y <= this.length) : abs(p.y) <= 10);
  }

  gizmoRect(index) {
    // index: 0 = rotate, 1 = toggle, 2 = move
    let lx = this.length / 2 + this.gizmoMargin;
    let ly = (index - 1) * (this.gizmoSize + 4);
    let center = this.screenFromLocal(lx, ly);

    return {
      x: center.x - this.gizmoSize / 2,
      y: center.y - this.gizmoSize / 2,
      w: this.gizmoSize,
      h: this.gizmoSize
    };
  }

  isMouseOverGizmo(index) {
    let r = this.gizmoRect(index);
    return mouseX >= r.x && mouseX <= r.x + r.w &&
           mouseY >= r.y && mouseY <= r.y + r.h;
  }

  isMouseOverAnyGizmo() {
    return this.isMouseOverGizmo(0) ||
           this.isMouseOverGizmo(1) ||
           this.isMouseOverGizmo(2);
  }

  // --- interaction ---

  updateHover() {
    this.hover = this.isMouseOverTrack() || this.isMouseOverAnyGizmo();
  }

  updateValueFromMouse() {
    let p = this.localFromScreen(mouseX, mouseY);
    let t = map(p.x, -this.length / 2, this.length / 2, -1, 1);
    this.v = constrain(t, -1, 1);
  }

  handleMousePressed() {
    this.updateHover();

    if (!this.hover) return;

    // 1. ROTATE GIZMO
    if (this.isMouseOverGizmo(0)) {
      this.draggingRotate = true;
      
      // Calculate the angle from the center of the slider to the mouse
      let dx = mouseX - this.x;
      let dy = mouseY - this.y;
      let mouseAngle = atan2(dy, dx);
      
      // Store the difference. 
      // This ensures that if you grab the handle slightly off-center, 
      // the slider doesn't "snap" to align perfectly with the mouse immediately.
      this.dragAngleOffset = this.angle - mouseAngle;
      return;
    }

    // 2. TOGGLE GIZMO
    if (this.isMouseOverGizmo(1)) {
      this.enabled = !this.enabled;
      return;
    }

    // 3. MOVE GIZMO
    if (this.isMouseOverGizmo(2)) {
      this.draggingMove = true;
      this.dragOffsetX = mouseX - this.x;
      this.dragOffsetY = mouseY - this.y;
      return;
    }

    // 4. TRACK DRAG
    if (this.isMouseOverTrack()) {
      this.draggingThumb = true;
      this.updateValueFromMouse();
    }
  }

  handleMouseDragged() {
    if (this.draggingThumb) {
      this.updateValueFromMouse();
      
    } else if (this.draggingRotate) {
      // Calculate new mouse angle relative to slider center
      let dx = mouseX - this.x;
      let dy = mouseY - this.y;
      let mouseAngle = atan2(dy, dx);
      
      // Apply angle with the initial offset preserved
      this.angle = mouseAngle + this.dragAngleOffset;
      
    } else if (this.draggingMove) {
      this.x = mouseX - this.dragOffsetX;
      this.y = mouseY - this.dragOffsetY;
    }
  }

  handleMouseReleased() {
    this.draggingThumb = false;
    this.draggingMove = false;
    this.draggingRotate = false;
  }

  // --- drawing ---

  drawTrackAndThumb() {
    push();
    translate(this.x, this.y);
    rotate(this.angle);

    if (!this.enabled) {
      rectMode(CENTER);
      fill(180, 100);
      noStroke();
      rect(0, this.length / 2, this.length, this.length, 4);
    }

    // track
    strokeWeight(this.thickness);
    stroke(this.enabled ? 220 : 120);
    line(-this.length / 2, 0, this.length / 2, 0);

    // thumb
    let tx = map(this.v, -1, 1, -this.length / 2, this.length / 2);
    rectMode(CENTER);
    stroke(0, 80);
    strokeWeight(1);
    let thumbFill = this.enabled ? (this.hover ? 255 : 235) : 160;
    fill(thumbFill);
    rect(tx, 0, 16, 26, 4);

    pop();
  }

  drawGizmos() {
    if (!this.hover && !this.draggingRotate && !this.draggingMove) return;

    textAlign(CENTER, CENTER);
    textSize(10);

    // Rotate gizmo (changed icon to circle arrow to indicate free rotation)
    this.drawGizmo(0, "↻"); 

    // Toggle gizmo
    this.drawGizmo(1, this.enabled ? "●" : "○");

    // Move gizmo
    this.drawGizmo(2, "✥");
  }

  drawGizmo(index, label) {
    let r = this.gizmoRect(index);
    let hovered = this.isMouseOverGizmo(index);
    
    // Visual feedback if we are currently dragging this specific operation
    let active = (index === 0 && this.draggingRotate) || 
                 (index === 2 && this.draggingMove);

    stroke(200);
    strokeWeight(1);
    fill(hovered || active ? 255 : 230, hovered || active ? 255 : 220);
    rect(r.x, r.y, r.w, r.h, 4);

    noStroke();
    fill(20);
    text(label, r.x + r.w / 2, r.y + r.h / 2);
  }

  draw() {
    this.updateHover();
    this.drawTrackAndThumb();
    this.drawGizmos();
  }
}