class ChatLine {
  constructor(w, h, onSend) {
    this.w = w;
    this.h = h;
    this.onSend = onSend;
    
    this.input = createInput('');
    this.input.size(w - 80, h); 
    this.input.style('font-size', '16px');
    this.input.style('border-radius', '8px');
    this.input.style('border-color', 'transparent');
    
    // Handle Enter key
    this.input.elt.addEventListener("keypress", (event) => {
      if (event.key === "Enter") {
        this.submit();
      }
    });

    this.sendButton = new SimpleButton();
  }

  draw(x, y, label = "Send") {
    // Button position
    this.input.position(x, y);
    const btnX = x + this.w - 70;
    const btnY = y;
    const btnW = 70;
    const btnH = this.h + 6; // Match input height roughly
    
    this.sendButton.draw(label, btnX, btnY, btnW, btnH);
  }

  mousePressed() 
  {
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
