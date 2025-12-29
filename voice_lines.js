const voiceLines = {};

function preloadVoiceLines() {
	voiceLines.command = ["great job", "I got you", "I'm here to help you"]
	.map(line => loadSound(`voice lines/${line}.wav`));
}

function playVoiceLine(category) {
  if (voiceLines[category] && voiceLines[category].length > 0) {
    const randomIndex = Math.floor(Math.random() * voiceLines[category].length);
    const sound = voiceLines[category][randomIndex];
    if (sound.isPlaying()) {
      sound.stop();
    }
    sound.play();
  }
}