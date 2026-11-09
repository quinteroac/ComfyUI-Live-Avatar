Live Avatar package

Run it
  python serve.py
Then open http://127.0.0.1:8765/ . Pass another port as an argument if that one is taken.

OBS
  Add a Browser source with this URL, size 720 x 1080:
  http://127.0.0.1:8765/?bg=transparent&ui=0
  Keep serve.py running. The background is transparent.

URL options, joined with &
  bg=transparent   no background, or a color such as bg=%23202020
  ui=0             hide the buttons
  frame=head       start on the head instead of the whole body
  clip=talk        loop a clip: wave, nod, shake, happy, think, talk, walk
  look=0           do not follow the mouse
  camera=1         start webcam face tracking
  mic=1            start lip sync from the microphone
  expression=smile start with an expression: smile, surprise, angry, sad, wink, sleepy, shy
  light=0.4,0.5    turn on shading with the light at x,y in -1..1, optional strength third
Keys 1 to 8 switch expressions while the page has focus.

Your own three.js app
  Copy runtime, three, avatar.lav and, for webcam tracking, mediapipe.
  Your import map needs "three" and "three/addons/postprocessing/Pass.js",
  as in viewer/index.html. Create the renderer with { stencil: true, alpha: true }.

    import { LiveAvatar } from "./runtime/live-avatar.mjs";
    const avatar = await LiveAvatar.load("avatar.lav", renderer);
    scene.add(avatar);
    avatar.play("idle");
    // every frame:
    avatar.update(deltaSeconds);

  avatar.play(name), avatar.stop(name), avatar.setParam(name, value) and
  avatar.lookAt(x, y), avatar.setExpression(name), avatar.setMouth(level, vowel),
  avatar.speak(text) and avatar.setLight({ x, y, strength }) drive it. avatar.params and avatar.clips list what it has.
