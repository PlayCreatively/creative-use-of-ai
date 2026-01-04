class SettingsGUI {
  constructor() {
    // Initialize settings state here to make them accessible
    this.generalSettings = [
        { label: 'Show Notifications', value: true },
        { label: 'Auto-Save', value: false }
    ];

    this.legacySettings = [
        { label: 'Legacy UI', value: false },
        { label: 'Legacy Move Handle', value: false },
        { label: 'Legacy Rotation Handle', value: false },
        { label: 'Legacy Trackpad Gizmo', value: false }
    ];

    this.createOverlay();
    this.createModal();
  }

  getValue(label) {
    const allSettings = [...this.generalSettings, ...this.legacySettings];
    const setting = allSettings.find(s => s.label === label);
    return setting ? setting.value : false;
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
    this.modalTitle.addClass('modal-title');
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
    
    this.modalOverlay.addClass('open');
  }

  closeModal() {
    this.modalOverlay.removeClass('open');
  }

  openHelpModal()
  {
    this.openModal('Command Cheat Sheet', (container) => {
          // Create a grid layout to use horizontal space
          let grid = createDiv('').parent(container);
          grid.addClass('cheat-sheet-grid');

          const addSection = (title, content) => {
            let section = createDiv('').parent(grid);
            createElement('h3', title).parent(section).addClass('cheat-sheet-title');
            createDiv(content).parent(section).addClass('cheat-sheet-content');
          };

          addSection('1. Actions <div class="mirrored-symbol">🗨</div>', 
            '<div class="actions-grid">' +
            '<div class="text-right">move</div><div class="symbol">✥</div><div class="text-right">rotate</div><div class="symbol">↻</div><div class="text-right">scale</div><div class="symbol">⤢</div>' +
            '<div class="text-right">stretch</div><div class="symbol">↔</div><div class="text-right">squash</div><div class="symbol">⇥⇤</div><div class="text-right">flip</div><div class="symbol">⇄</div>' +
            '<div class="text-right">shear</div><div class="symbol">▱</div>' +
            '</div>');

          addSection('2. Axis Transformations <div class="mirrored-symbol">🗨</div>', 
            '<div class="axis-grid">' +
            '<div class="text-right">x</div><div class="symbol">↔</div><div class="text-right">y</div><div class="symbol">↕</div>' +
            '<div class="text-right">left</div><div class="symbol">←</div><div class="text-right">right</div><div class="symbol">→</div>' +
            '<div class="text-right">up</div><div class="symbol">↑</div><div class="text-right">down</div><div class="symbol">↓</div>' +
            '</div>');

          addSection('3. Intensity <div class="mirrored-symbol">🗨</div>', 
            '<div class="intensity-grid">' +
            '<div class="symbol">▲</div><div><b>Increase:</b></div><div>lot, very, extremely, much</div>' +
            '<div class="symbol">▼</div><div><b>Decrease:</b></div><div>little, slightly, bit, less</div>' +
            '</div>');
        });
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
    this.createDropdown('File - IMPLEMENT', [
      { label: 'New Project', action: () => console.log('New Project') },
      { label: 'Save', action: () => console.log('Save') },
      { label: 'Load', action: () => console.log('Load') }
    ]);

    // Edit Menu
    this.createDropdown('Edit - IMPLEMENT', [
      { label: 'Undo', action: () => console.log('Undo') },
      { label: 'Redo', action: () => console.log('Redo') },
      { label: 'Clear Canvas', action: () => console.log('Clear Canvas') }
    ]);
    
    // Settings Menu
    this.createDropdown('Settings', [
      { 
        label: 'Open Settings', 
        action: () => this.openModal('Settings', (container) => {
            // --- General Settings ---
            let settingsHeader = createElement('h3', 'General Settings');
            settingsHeader.parent(container);
            settingsHeader.addClass('settings-header');

            // Helper function to create toggles
            const toggleSetting = (settings) => {
              settings.forEach(setting => {
                  let row = createDiv('');
                  row.parent(container);
                  row.addClass('settings-row');

                  // Checkbox
                  let chk = createCheckbox('', setting.value);
                  chk.parent(row);
                  chk.addClass('settings-checkbox');
                  
                  // Update the setting object when changed
                  chk.changed(() => {
                    setting.value = chk.checked();
                    const isAnyLegacy = setting.label === 'Legacy UI' || 
                                        setting.label === 'Legacy Move Handle' ||
                                        setting.label === 'Legacy Rotation Handle' ||
                                        setting.label === 'Legacy Trackpad Gizmo';

                    if(setting.value && isAnyLegacy)
                      playVoiceLine('warning');
                  });
                  
                  // Label
                  let lbl = createSpan(setting.label);
                  lbl.parent(row);
              });
            }

            toggleSetting(this.generalSettings);

            // --- Legacy Section ---
            let legacyHeader = createElement('h3', 'Legacy Features');
            legacyHeader.parent(container);
            legacyHeader.addClass('settings-header');
            legacyHeader.addClass('legacy-header');
            
            let warning = createP('⚠️ Warning: These features are not meant for the general public and are no longer maintained. <br/>Use at your own risk.');
            warning.parent(container);
            warning.addClass('warning-box');
            
            toggleSetting(this.legacySettings);
        }) 
      }
    ]);
    
    // Help Menu
    this.createDropdown('Help', [
      { label: 'How To', action: () => {
        this.openHelpModal();
      }},
      { label: 'About', action: () => alert('Creative AI Project v1.0') }
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
        if (item.action) item.action();
      });
    });
  }
}
