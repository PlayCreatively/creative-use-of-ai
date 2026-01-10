const MOVE_GIZMO = 2;
const TRACKPAD_GIZMO = 1;
const ROTATE_GIZMO = 0;

class FancySlider {
  constructor(x, y, length, angle = 0, color = null) {
    this.x = x;
    this.y = y;
    this.length = length;
    this.thickness = 6;
    this.color = color || ((typeof window.color === 'function') ? window.color(200) : { levels: [200,200,200,255] }); // Fallback safe check

    this.angle = angle;       
    this.v = 0.0;             // -1..1
    this.trackpadActive = true;      

    this.hover = false;
    
    // Interaction flags
    this.draggingThumb = false;
    this.draggingMove = false;
    this.draggingRotate = false;
    
    // Offsets to keep interaction smooth
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;
    this.dragAngleOffset = 0;    // offset to prevent snapping

    this.gizmoSize = 22;
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
           (!this.trackpadActive ? (p.y > 0 && p.y <= this.length) : abs(p.y) <= 10);
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

  updateHover(isSelected) {
    this.hover = this.isMouseOverTrack() || (this.isMouseOverAnyGizmo() && isSelected);
  }

  updateValueFromMouse() {
    let p = this.localFromScreen(mouseX, mouseY);
    let t = map(p.x, -this.length / 2, this.length / 2, -1, 1);
    this.v = constrain(t, -1, 1);
  }

  handleMousePressed(isSelected) {
    this.updateHover(isSelected);
    
    if (!this.hover || !this.enabled) return false;
    
    // TRACK DRAG
    if (this.isMouseOverTrack()) {
      this.draggingThumb = true;
      this.updateValueFromMouse();
    }
    
    // ROTATE GIZMO
    else if (this.rotationEnabled && this.isMouseOverGizmo(0)) {
      this.draggingRotate = true;
      
      // Calculate the angle from the center of the slider to the mouse
      let dx = mouseX - this.x;
      let dy = mouseY - this.y;
      let mouseAngle = atan2(dy, dx);
      
      // Store the difference. 
      // This ensures that if you grab the handle slightly off-center, 
      // the slider doesn't "snap" to align perfectly with the mouse immediately.
      this.dragAngleOffset = this.angle - mouseAngle;
    }

    // TRACKPAD GIZMO
    else if (this.trackpadEnabled && this.isMouseOverGizmo(1)) {
      this.trackpadActive = !this.trackpadActive;
    }

    // MOVE GIZMO
    else if (this.moveEnabled && this.isMouseOverGizmo(2)) {
      this.draggingMove = true;
      this.dragOffsetX = mouseX - this.x;
      this.dragOffsetY = mouseY - this.y;
    }

    return true;
  }

  handleMouseDragged(isSelected) {
    
    if(!this.enabled) return false;

    if (this.draggingThumb) {
      this.updateValueFromMouse();
      
    } else if (!isSelected) return false;

    else if (this.rotationEnabled && this.draggingRotate) {
      // Calculate new mouse angle relative to slider center
      let dx = mouseX - this.x;
      let dy = mouseY - this.y;
      let mouseAngle = atan2(dy, dx);
      
      // Apply angle with the initial offset preserved
      this.angle = mouseAngle + this.dragAngleOffset;
      
    } else if (this.moveEnabled && this.draggingMove) {
      this.x = mouseX - this.dragOffsetX;
      this.y = mouseY - this.dragOffsetY;

    } else return false;
    return true;
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

    let tx = map(this.v, -1, 1, -this.length / 2, this.length / 2);

    if (!this.trackpadActive) {
      rectMode(CENTER);
      fill(180, 100);
      noStroke();
      rect(0, this.length / 2, this.length, this.length, 4);

      // Trackpad line
      stroke(this.color);
      strokeWeight(2);
      line(tx, 0, tx, this.length);
    }

    // track
    strokeWeight(this.thickness);
    if (this.trackpadActive) {
      stroke(this.color);
    } else {
      let c = this.color;
      stroke(red(c), green(c), blue(c), 120);
    }
    line(-this.length / 2, 0, this.length / 2, 0);

    // thumb
    rectMode(CENTER);
    stroke(0, 80);
    strokeWeight(1);
    let thumbFill = this.trackpadActive ? (this.hover ? 255 : 235) : 160;
    fill(thumbFill);
    rect(tx, 0, 16, 26, 4);

    pop();
  }

  updateEnableStates(enabled) {
    [this.enabled, this.moveEnabled, this.rotationEnabled, this.trackpadEnabled] = enabled;
  }

  drawGizmos() {
    if(!this.enabled) return;

    push();

    textAlign(CENTER, CENTER);
    textSize(16);

    // Rotate gizmo (changed icon to circle arrow to indicate free rotation)
    if(this.rotationEnabled)
      this.drawGizmo(ROTATE_GIZMO, "↻"); 
    
    // Move gizmo
    if(this.moveEnabled)
      this.drawGizmo(MOVE_GIZMO, "✥");

    // Trackpad gizmo
    if(this.trackpadEnabled)
      this.drawGizmo(TRACKPAD_GIZMO, this.trackpadActive ? "●" : "○");

    pop();
  }

  drawGizmo(index, label) {
    let r = this.gizmoRect(index);
    let hovered = this.isMouseOverGizmo(index);
    
    // Visual feedback if we are currently dragging this specific operation
    let active = (index === ROTATE_GIZMO && this.draggingRotate) || 
                 (index === MOVE_GIZMO && this.draggingMove);

    stroke(200);
    strokeWeight(1);
    fill(hovered || active ? 255 : 230, hovered || active ? 255 : 220);
    rect(r.x, r.y, r.w, r.h, 4);

    noStroke();
    fill(20);
    text(label, r.x + r.w / 2, r.y + r.h / 2);
  }

  draw() {
    if(!this.enabled) return;
    
    this.updateHover();
    this.drawTrackAndThumb();
  }

  static saveAll(sliders) {
    const sliderData = sliders.map(s => ({
      x: s.x,
      y: s.y,
      angle: s.angle,
      v: s.v,
      trackpadActive: s.trackpadActive
    }));
    localStorage.setItem('sliderData', JSON.stringify(sliderData));
  }

  static loadAll(sliders) {
    const data = localStorage.getItem('sliderData');
    if (data) {
      const sliderData = JSON.parse(data);
      for (let i = 0; i < min(sliders.length, sliderData.length); i++) {
        sliders[i].x = sliderData[i].x;
        sliders[i].y = sliderData[i].y;
        sliders[i].angle = sliderData[i].angle;
        sliders[i].v = sliderData[i].v;
        sliders[i].trackpadActive = sliderData[i].trackpadActive;
      }
    }
  }
}