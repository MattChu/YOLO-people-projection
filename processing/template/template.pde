// template - a starting point for a floor piece driven by the tracker.
// Copy this folder to processing/<name>/ and rename template.pde to <name>.pde.
//
// Run the tracker alongside it:
//   python track_osc.py               # people, from the camera
//   python track_osc.py --simulate 3  # three imaginary people, no camera
// or hold the mouse button down for a test person at the mouse.
//
// People tab: who is on the floor (`people`, positions in metres, steadied
// and faded in and out), with personArrived() / personLeft() below called as
// they come and go. Copy it unchanged to every sketch.
// Projection tab: draw into `canvas`, which covers the taped rectangle
// (PIXELS_PER_METRE pixels per metre, origin at the first tape mark); it is
// warped onto the floor using calibration.json.

void settings() {
  // full screen on the projector, or a window when it isn't connected (Projection tab)
  chooseScreen();
}

void setup() {
  frameRate(60);
  setupProjection();
  startPeopleReceiver();
}

void draw() {
  updatePeople();

  canvas.beginDraw();
  canvas.background(0);

  // the taped rectangle's edge, to check the calibration lines up with the tape
  canvas.noFill();
  canvas.stroke(80);
  canvas.strokeWeight(4);
  canvas.rect(2, 2, canvas.width - 4, canvas.height - 4);

  // a circle at everyone's feet, 30 cm across, fading in and out
  canvas.noStroke();
  for (Person p : people) {
    if (p.shown == 0) continue;
    canvas.fill(255, 255 * p.shown);
    canvas.ellipse(p.canvasX(), p.canvasY(), 0.6 * PIXELS_PER_METRE, 0.6 * PIXELS_PER_METRE);
  }
  canvas.endDraw();

  // warp the canvas onto the taped area (Projection tab)
  drawProjected(canvas);
}

// someone new has stepped onto the floor (People tab)
void personArrived(Person p) {
  println("arrived: #" + p.id + " at " + nf(p.x, 0, 2) + ", " + nf(p.y, 0, 2) + " m");
}

// someone has gone: lost, faded out and not found again within REATTACH_MS
// (People tab). While lost they were still in `people`, with isLost() true
void personLeft(Person p) {
  println("left: #" + p.id);
}
