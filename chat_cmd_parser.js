let lastCommand = "";

function parseCommand(command, M) {

  command = command.toLowerCase().trim();

  if (lastCommand)
    if (stringContainsAny(command, ["again", "repeat"]) || command === "more")
      command = lastCommand;

  // ----- Determine multiplier ----- //

  const increase = ["lot", "very", "extremely", "more", "bigger", "further", "much", "far"];
  const decrease = ["little", "slightly", "less", "smaller", "reduce", "bit"];

  const multiplier = stringContainsAny(command, increase) ? 2 : stringContainsAny(command, decrease) ? 0.5 : 1.0;

  // ----- Translation ----- //

  const moveCmd = ["move", "go", "translate", "shift"];
  const moveRightCmd = ["right", "east"];
  const moveLeftCmd = ["left", "west"];
  const moveUpCmd = ["up", "north"];
  const moveDownCmd = ["down", "south"];

  const translationX = (stringContainsAny(command, moveCmd) ? 1 : 0) 
  * ((stringContainsAny(command, moveRightCmd) ? 1 : 0) + (stringContainsAny(command, moveLeftCmd) ? -1 : 0)) * multiplier * 0.25;

  const translationY = (stringContainsAny(command, moveCmd) ? 1 : 0) 
  * ((stringContainsAny(command, moveDownCmd) ? 1 : 0) + (stringContainsAny(command, moveUpCmd) ? -1 : 0)) * multiplier * 0.25;

  // ----- Rotation ----- //

  const rotateCmd = ["rotate", "turn", "spin", "twist"];
  const rotateLeftCmd = ["left", "counterclockwise"];
  const rotateRightCmd = ["right", "clockwise"];

  const rotation = (stringContainsAny(command, rotateCmd) ? 1 : 0) 
  * (stringContainsAny(command, rotateLeftCmd) ? -1 : 1) * multiplier * Math.PI * 0.25;


  // ----- Scaling ----- //

  const scaleUp = ["scale up", "increase size", "enlarge", "bigger"];
  const scaleDown = ["scale down", "decrease size", "shrink", "smaller"];
  
  const scale = ((stringContainsAny(command, scaleUp) ? 1 : 0) 
  + (stringContainsAny(command, scaleDown) ? -1 : 0)) * multiplier * 0.5;

  // ----- Axis scale ----- //

  const axisScaleUpCmd = ["stretch", "scale", "widen", "expand"];
  const axisScaleDownCmd = ["shrink", "squash", "scale down", "compress"];
  const axisXCmd = [" x", " x-", "horizontal", "width", " w", "right", "left"];
  const axisYCmd = [" y", " y-", "vertical", "height", " h"];

  const hasX = stringContainsAny(command, axisXCmd);
  const hasY = stringContainsAny(command, axisYCmd);

  // If targeting an axis, ignore global scale to prevent side effects on the other axis
  let xScale = 1 + ((hasX || hasY) ? 0 : scale);
  let yScale = 1 + ((hasX || hasY) ? 0 : scale);

  const axisScaleCmd = (stringContainsAny(command, axisScaleDownCmd) ? -1 : (stringContainsAny(command, axisScaleUpCmd) ? 1 : 0));

  if (axisScaleCmd != 0) 
  {
    if (hasX)
      xScale = 1 + axisScaleCmd * .25 * multiplier;

    if (hasY)
      yScale = 1 + axisScaleCmd * .25 * multiplier;
  }

  // ----- Flip ----- //

  const flipCmd = ["flip", "mirror", "invert", "upside down"];
  const flipYAxisCmd = [" y", " y-", "vertical", "upside down"];
  const flipXAxisCmd = [" x", " x-", "horizontal", "flip"];

  if (stringContainsAny(command, flipCmd)) 
  {
    if (stringContainsAny(command, flipYAxisCmd))
      yScale *= -1;
    
    if (stringContainsAny(command, flipXAxisCmd)) 
      xScale *= -1;
  }

  const shearCmd = ["shear", "skew"];
  const shearXCmd = [" x", " x-", "horizontal"];
  const shearYCmd = [" y", " y-", "vertical"];
  const shearPosCmd = [" positive", "+", "clockwise", "right", "up"];
  const shearNegCmd = [" negative", "-", "counterclockwise", "left", "down"];

  let shearX = 0;
  let shearY = 0;

  if (stringContainsAny(command, shearCmd)) 
  {
    if (stringContainsAny(command, shearXCmd))
      shearX = (stringContainsAny(command, shearNegCmd) ? -1 : 1) * 0.5 * multiplier;
    
    if (stringContainsAny(command, shearYCmd)) 
      shearY = (stringContainsAny(command, shearNegCmd) ? -1 : 1) * 0.5 * multiplier;
  }
  
  // ----- Build target matrix ----- //
  
  // Apply global translation
  let aiTargetMatrix = M.clone();
  let translationMatrix = getTransformMatrix(0, 1, 1, translationX, translationY);
  aiTargetMatrix = math.multiply(translationMatrix, aiTargetMatrix);
  
  // Apply shear
  if (shearX != 0 || shearY != 0) {
    // Build shear matrix and apply
    let shearM = math.matrix([
      [1, -shearX, 0],
      [-shearY, 1, 0],
      [0, 0, 1]
    ]);
    aiTargetMatrix = math.multiply(shearM, aiTargetMatrix);
  }

  // Combine to ensure Scale is applied before Rotation to avoid skewing
  let localMatrix = getTransformMatrix(rotation, xScale, yScale, 0, 0);
  aiTargetMatrix = math.multiply(aiTargetMatrix, localMatrix);

  if(math.deepEqual(aiTargetMatrix, M))
    return null; // no-op

//  alert("Applying transformation:\n" +
//         "Translation: (" + translationX.toFixed(2) + ", " + translationY.toFixed(2) + ")\n" +
//         "Rotation: " + rotation.toFixed(2) + " turns\n" +
//         "Scale: " + scale.toFixed(2));

  lastCommand = command;

  return aiTargetMatrix;
}