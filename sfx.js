let sfx = {};

function preloadSFX()
{
	soundFormats('wav');
	sfx.confirm = loadSound('sfx/small tada.wav');
	sfx.update = loadSound('sfx/CHORD.wav');
	sfx.privacyCorrection = loadSound('sfx/Windows XP Battery Low.wav');
	sfx.submit = loadSound('sfx/TADA.wav');
	sfx.openModal = loadSound('sfx/CHIMES.wav');
}