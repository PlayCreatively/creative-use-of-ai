class SettingsGUI {
  constructor() {
    this.createOverlay();
    this.createModal();
  }

  createModal() {
    // Create Overlay
    this.modalOverlay = createDiv('');
    this.modalOverlay.class('modal-overlay');
    this.modalOverlay.parent(document.body);
    
    // Close when clicking outside
    this.modalOverlay.mousePressed((e) => {
        // Check if the click is on the overlay itself, not the content
        if (e.target === this.modalOverlay.elt) {
            this.closeModal();
        }
    });

    // Create Content Box
    let modalContent = createDiv('');
    modalContent.class('modal-content');
    modalContent.parent(this.modalOverlay);

    // Close Button
    let closeBtn = createSpan('&times;');
    closeBtn.class('close-btn');
    closeBtn.parent(modalContent);
    closeBtn.mousePressed(() => this.closeModal());

    // Title
    this.modalTitle = createElement('h2', '');
    this.modalTitle.style('margin-top', '0');
    this.modalTitle.parent(modalContent);

    // Body Container
    this.modalBody = createDiv('');
    this.modalBody.parent(modalContent);
  }

  openModal(title, contentBuilder) {
    this.modalTitle.html(title);
    this.modalBody.html(''); // Clear previous content
    
    if (contentBuilder) {
        contentBuilder(this.modalBody);
    }
    
    this.modalOverlay.style('display', 'flex');
  }

  closeModal() {
    this.modalOverlay.style('display', 'none');
  }

  createOverlay() {
    // Create the main top bar container
    // We use p5.js createDiv
    this.topBar = createDiv('');
    this.topBar.id('top-bar');
    // Ensure it is attached to body so it overlays everything
    this.topBar.parent(document.body);

    // --- Example Menu Items ---
    
    // File Menu
    this.createDropdown('File', [
      { label: 'New Project', action: () => console.log('New Project') },
      { label: 'Save', action: () => console.log('Save') },
      { label: 'Load', action: () => console.log('Load') }
    ]);

    // Edit Menu
    this.createDropdown('Edit', [
      { label: 'Undo', action: () => console.log('Undo') },
      { label: 'Redo', action: () => console.log('Redo') },
      { label: 'Clear Canvas', action: () => console.log('Clear Canvas') }
    ]);
    
    // Help Menu
    this.createDropdown('Help', [
      { label: 'About', action: () => alert('Creative AI Project v1.0') }
    ]);
    
    // Settings Menu
    this.createDropdown('Settings', [
      { 
        label: 'Open Settings', 
        action: () => this.openModal('Settings', (container) => {
            // Legacy Content
            let legacyInfo = createP('Legacy: Creative AI Project v1.0');
            legacyInfo.parent(container);

            // Image
            let img = createImg('images/we can do it [Gemini Generated].jpeg', 'We can do it');
            img.parent(container);
            img.style('display', 'block');
            img.style('width', '50%');
            img.style('margin', '10px auto');
            img.style('border-radius', '5px');
        }) 
      }
    ]);
  }

  createDropdown(label, items) {
    // Container for the menu item (button + dropdown list)
    let menuContainer = createDiv('');
    menuContainer.class('menu-item');
    menuContainer.parent(this.topBar);

    // The visible button on the bar
    let btn = createButton(label);
    btn.class('menu-btn');
    btn.parent(menuContainer);

    // The dropdown content container (hidden by default via CSS)
    let content = createDiv('');
    content.class('dropdown-content');
    content.parent(menuContainer);

    // Add items to the dropdown
    items.forEach(item => {
      let itemBtn = createButton(item.label);
      itemBtn.parent(content);
      itemBtn.mousePressed(() => {
        // Execute the action
        if (item.action) item.action();
        
        // Optional: Close logic is handled by CSS hover, 
        // but if we wanted click-toggle, we'd do it here.
      });
    });
  }
}
