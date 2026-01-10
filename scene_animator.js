
class SceneAnimator {
  constructor() {
    this.mode = 'idle'; // 'idle', 'flyOut', 'panIn'
    this.startT = 0;
    this.dur = 1000;
    this.cb = null;
  }
  
  submit(callback) {
      this.mode = 'flyOut';
      this.startT = millis();
      this.dur = 1000;
      this.cb = callback;
  }
  
  intro() {
      this.mode = 'panIn';
      this.startT = millis();
      this.dur = 1200;
  }
  
  // Handle the "Fly Out" animation loop
  // Returns true if animation is playing and handled
  // Returns false if we should show the static UI
  handleFlyOut(library, height) {
      if (this.mode !== 'flyOut') return false;
      
      let t = (millis() - this.startT) / this.dur;
      
      let ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; 
      translate(0, -height * ease);
      
      if (t >= 1) {
          if (this.cb) this.cb();
          // Expect cb to trigger intro or reset
      }
      
      return true;
  }
  
  // Call `applyPanIn(width)` inside a push/pop block roughly wrapping the scene
  applyPanIn(width) {
      if (this.mode !== 'panIn') return;
      
      let t = (millis() - this.startT) / this.dur;
      if (t >= 1) {
          // ensure we finish cleanly
          this.mode = 'idle';
          return;
      }
      
      let ease = 1 - Math.pow(1 - t, 3); // cubic ease out
      let x = map(ease, 0, 1, -width, 0);
      translate(x, 0);
  }
}