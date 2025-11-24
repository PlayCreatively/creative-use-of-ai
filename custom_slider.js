class FancySlider {
  constructor(x, y, length, angle = 0) {
    this.x = x;
    this.y = y;
    this.length = length;
    this.thickness = 6;

    this.angle = angle;            // 0 or HALF_PI
    this.v = 0.0;          // -1..1
    this.enabled = true;       // now visual only, does NOT lock the slider

    this.hover = false;
    this.draggingThumb = false;
    this.draggingMove = false;
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;

    this.gizmoSize = 18;
    this.gizmoMargin = 32;     // increased margin
  }

  value(newValue) {
	if (newValue === undefined) {
    return this.v;
  }
  this.v = constrain(newValue, -1, 1); // optional: clamp
  return this.v;
}

  // --- coordinate helpers (local slider space vs screen) ---

  screenFromLocal(px, py) {
    // rotate (px, py) around (this.x, this.y) by this.angle
    let c = cos(this.angle);
    let s = sin(this.angle);
    let sx = this.x + px * c - py * s;
    let sy = this.y + px * s + py * c;
    return createVector(sx, sy);
  }

  localFromScreen(mx, my) {
    // inverse rotation: screen -> slider local
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
    // Treat slider as a thin rectangle along x-axis
    return p.x >= -this.length / 2 &&
           p.x <=  this.length / 2 &&
           abs(p.y) <= 10;
  }

  gizmoRect(index) {
    // index: 0 = rotate, 1 = toggle, 2 = move
    // Place gizmos in *local* space next to the +X end of the slider,
    // then rotate into screen space.
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

    // Gizmos take priority
    if (this.isMouseOverGizmo(0)) {
      // rotate
      this.angle = (this.angle + HALF_PI) % TWO_PI;
      return;
    }

    if (this.isMouseOverGizmo(1)) {
      // toggle (visual state only now)
      this.enabled = !this.enabled;
      return;
    }

    if (this.isMouseOverGizmo(2)) {
      // move whole slider
      this.draggingMove = true;
      this.dragOffsetX = mouseX - this.x;
      this.dragOffsetY = mouseY - this.y;
      return;
    }

    // Drag thumb — no longer blocked by enabled/disabled
    if (this.isMouseOverTrack()) {
      this.draggingThumb = true;
      this.updateValueFromMouse();
    }
  }

  handleMouseDragged() {
    if (this.draggingThumb) {
      this.updateValueFromMouse();
    } else if (this.draggingMove) {
      this.x = mouseX - this.dragOffsetX;
      this.y = mouseY - this.dragOffsetY;
    }
  }

  handleMouseReleased() {
    this.draggingThumb = false;
    this.draggingMove = false;
  }

  // --- drawing ---

  drawTrackAndThumb() {
    push();
    translate(this.x, this.y);
    rotate(this.angle);

    // track
    strokeWeight(this.thickness);
    // still dim track when "disabled", but purely cosmetic now
    stroke(this.enabled ? 220 : 120);
    line(-this.length / 2, 0, this.length / 2, 0);

    // thumb position in local coords
    let tx = map(this.v, -1, 1, -this.length / 2, this.length / 2);

    // thumb
    rectMode(CENTER);
    stroke(0, 80);
    strokeWeight(1);
    let thumbFill = this.enabled ? (this.hover ? 255 : 235) : 160;
    fill(thumbFill);
    rect(tx, 0, 16, 26, 4);

    pop();
  }

  drawGizmos() {
    if (!this.hover) return;

    textAlign(CENTER, CENTER);
    textSize(10);

    // Rotate gizmo
    this.drawGizmo(0, "⤾");

    // Toggle gizmo
    this.drawGizmo(1, this.enabled ? "●" : "○");

    // Move gizmo
    this.drawGizmo(2, "✥");
  }

  drawGizmo(index, label) {
    let r = this.gizmoRect(index);
    let hovered = this.isMouseOverGizmo(index);

    stroke(200);
    strokeWeight(1);
    fill(hovered ? 255 : 230, hovered ? 255 : 220);
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
