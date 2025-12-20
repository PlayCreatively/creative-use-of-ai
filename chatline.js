class ChatLine {
  constructor(x, y, w, h, onSend) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.onSend = onSend;
    
    this.input = createInput('');
    this.input.position(x, y);
    this.input.size(w - 80, h); 
    this.input.style('font-size', '16px');
    
    // Handle Enter key
    this.input.elt.addEventListener("keypress", (event) => {
      if (event.key === "Enter") {
        this.submit();
      }
    });

    this.sendButton = new SimpleButton();
  }

  draw(label = "Send") {
    // Button position
    let btnX = this.x + this.w - 70;
    let btnY = this.y;
    let btnW = 70;
    let btnH = this.h + 6; // Match input height roughly
    
    this.sendButton.draw(label, btnX, btnY, btnW, btnH);
  }

  mousePressed() {
    if (this.sendButton.isHovering(mouseX, mouseY)) {
      this.submit();
    }
  }

  submit() {
    let msg = this.input.value();
    
      if (this.onSend) {
        this.onSend(msg);
      
      this.input.value('');
    }
  }
}
