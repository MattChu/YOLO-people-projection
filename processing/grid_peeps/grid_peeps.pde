// grid_peeps - a horizontal and a vertical line through everyone on the floor,
// in their colour.
// Each person has a colour. Every pair of people spans a rectangle (they are
// at opposite corners), where copies of the logo (data/mill_logo.svg) swirl,
// each a different size and a different shade between the pair's two colours.
// Where rectangles overlap, so do their swirls.
//
// TRAIL mode (press m to switch): no lines, but everyone leaves a jumbled
// trail of the logo as they walk, in their colour: copies of all sizes,
// overlapping, each circling and turning as it fades away.
//
// FIELD mode: the floor is covered in copies of the logo, swirling on their
// spots. They part around anyone walking through, and slowly drift back
// together behind them: the opposite of the trail.
//
// In all three, some copies are whole and some broken into their parts (Logo
// tab), which swirl and turn on their own.
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

import java.util.HashMap;
import java.util.Random;

// which piece shows: GRID (the lines, and logos swirling between people),
// TRAIL (everyone leaves a trail of logos) or FIELD (a floor full of logos
// that part around people). Press m to go to the next while it runs
final int GRID = 0, TRAIL = 1, FIELD = 2;
final String[] MODE_NAMES = {"grid", "trail", "field"};
int mode = FIELD;

// people's colours, yellow-green through teal to violet-blue, about 30 degrees
// of hue apart and all near full brightness (a dim colour on the projector
// looks like a faded person): a newcomer gets the one fewest people have now
color[] PALETTE = {
  color(150, 255, 0),   // chartreuse
  color(0, 235, 60),    // green
  color(0, 235, 165),   // sea green
  color(0, 215, 255),   // cyan
  color(0, 120, 255),   // azure
  color(80, 70, 255)    // indigo-blue
};

// true: the shades between two people's colours are mixed in linear light, as
// coloured lights mix, so they stay bright; false: plain RGB averages, which
// go darker and muddier between strong colours
final boolean BLEND_AS_LIGHT = true;

// the logo (in data/; Logo tab). Its lines are thickened to LOGO_LINE_M
// metres on a LOGO_IMAGE_M-wide copy, as at its own size they'd be a couple of
// millimetres on the floor
final String LOGO_FILE = "mill_logo.svg";
final float LOGO_LINE_M = 0.012;
// in all modes, this fraction of the copies are broken into their parts
// (the gear-and-house, and each letter), which swirl and turn on their own
final float BROKEN_CHANCE = 0.5;

// GRID: copies swirling in each rectangle
final int LOGOS_PER_BOX = 7;
// a copy's width: this fraction of its rectangle's shorter side, but at least
// LOGO_MIN_M metres, so a thin rectangle still shows recognisable logos;
// each copy is then scaled by a random amount between these
final float LOGO_SIZE = 0.6;
final float LOGO_MIN_M = 0.3;
final float GRID_SCALE_MIN = 0.5, GRID_SCALE_MAX = 1.3;
// how opaque each copy is (0-255): less lets the overlaps show through each other
final float LOGO_ALPHA = 190;
// how fast the copies go round, in turns a minute (each between these)
final float SWIRL_MIN = 2, SWIRL_MAX = 6;
// broken copies: how far their parts wander from their places, as a fraction
// of the copy's width, and how fast they turn (radians a second, at most)
final float GRID_SCATTER = 0.35;
final float GRID_PART_SPIN = 0.8;

// TRAIL: a copy is dropped every TRAIL_SPACING_M metres someone walks,
// somewhere within TRAIL_SCATTER_M of their path, so they pile up into a
// jumble; widths between these (more small ones than big)
final float TRAIL_SPACING_M = 0.06;
final float TRAIL_SCATTER_M = 0.12;
final float TRAIL_SIZE_MIN_M = 0.1, TRAIL_SIZE_MAX_M = 0.4;
// each one fades in, then away over TRAIL_LIFE_S seconds...
final float TRAIL_LIFE_S = 6;
final float TRAIL_ALPHA = 255;
// ...while it circles its spot up to TRAIL_SWIRL_M metres out, about once
// every TRAIL_SWIRL_S seconds, and turns at up to TRAIL_SPIN_DEG degrees a second
final float TRAIL_SWIRL_M = 0.08;
final float TRAIL_SWIRL_S = 6;
final float TRAIL_SPIN_DEG = 30;
// broken copies: as GRID_SCATTER and GRID_PART_SPIN
final float TRAIL_SCATTER = 0.5;
final float TRAIL_PART_SPIN = 1.2;
// at most this many on the floor (the oldest go first), to keep the frame rate up
final int TRAIL_MAX = 500;

// FIELD: about FIELD_PER_M2 copies a square metre, spread evenly over the
// floor, widths between these (more small ones than big), each circling and
// turning on its spot as in TRAIL
final float FIELD_PER_M2 = 45;
final float FIELD_SIZE_MIN_M = 0.1, FIELD_SIZE_MAX_M = 0.35;
final float FIELD_ALPHA = 200;
final float FIELD_SCATTER = 0.35;
final float FIELD_PART_SPIN = 0.8;
// people push them out to FIELD_CLEAR_M metres from their feet (plus a bit
// for big ones), quickly but not in one jump...
final float FIELD_CLEAR_M = 0.30;
final float FIELD_PUSH_RATE = 12;  // fraction of the way a second
// ...and once they've passed, the copies drift home, most of the way (95%)
// in FIELD_RETURN_S seconds
final float FIELD_RETURN_S = 4;
// broken copies come further apart while pushed: up to this much more scatter
final float FIELD_PUSH_SCATTER = 1.0;

ArrayList<FieldLogo> field = new ArrayList<FieldLogo>();
int lastFieldMillis = -1;
ArrayList<Stamp> trail = new ArrayList<Stamp>();
// where each person last dropped a copy
HashMap<Person, PVector> lastStamp = new HashMap<Person, PVector>();
HashMap<Person, Integer> colours = new HashMap<Person, Integer>();

void settings() {
  // full screen on the projector, or a window when it isn't connected (Projection tab)
  chooseScreen();
}

void setup() {
  frameRate(60);
  setupProjection();
  loadLogo();
  makeField();
  startPeopleReceiver();
}

void draw() {
  updatePeople();

  // everyone visible, fading in or out
  ArrayList<Person> shown = new ArrayList<Person>();
  for (Person p : people) {
    if (p.shown > 0) shown.add(p);
  }

  canvas.beginDraw();
  canvas.background(0);

  if (mode == TRAIL || mode == FIELD) {
    if (mode == TRAIL) {
      updateTrail();
      drawTrail();
    } else {
      updateField();
      drawField();
    }
    canvas.endDraw();
    drawProjected(canvas);
    return;
  }

  drawSwirls(shown);

  // the taped rectangle's edge, to check the calibration lines up with the tape
  canvas.noFill();
  canvas.stroke(80);
  canvas.strokeWeight(4);
  //canvas.rect(2, 2, canvas.width - 4, canvas.height - 4);

  // a horizontal and a vertical line through everyone, in their colour
  for (Person p : shown) {
    canvas.stroke(colourOf(p), 255 * p.shown);
    canvas.line(p.canvasX(), 0, p.canvasX(), canvas.height);
    canvas.line(0, p.canvasY(), canvas.width, p.canvasY());
  }
  canvas.endDraw();

  // warp the canvas onto the taped area (Projection tab)
  drawProjected(canvas);
}

// Copies of the logo swirling round the middle of each pair's rectangle,
// clipped to it, faded with the fainter of the two people.
void drawSwirls(ArrayList<Person> shown) {
  float t = millis() / 1000.0;
  for (int a = 0; a < shown.size(); a++) {
    for (int c = a + 1; c < shown.size(); c++) {
      Person p = shown.get(a), q = shown.get(c);
      float x0 = min(p.canvasX(), q.canvasX()), x1 = max(p.canvasX(), q.canvasX());
      float y0 = min(p.canvasY(), q.canvasY()), y1 = max(p.canvasY(), q.canvasY());
      float w = x1 - x0, h = y1 - y0;
      if (w < 1 || h < 1) continue;  // in line with each other: no rectangle
      float fade = min(p.shown, q.shown);
      color pc = colourOf(p), qc = colourOf(q);
      float logoW = max(LOGO_MIN_M * PIXELS_PER_METRE, LOGO_SIZE * min(w, h));

      // the same pair always swirls the same way (whoever arrived first), so
      // working it out afresh each frame gives smooth motion
      Random rnd = new Random(min(p.id, q.id) * 100003L + max(p.id, q.id));
      float direction = rnd.nextBoolean() ? 1 : -1;

      // (clip reads its corners by imageMode, so before it's set to CENTER)
      canvas.clip(x0, y0, w, h);
      canvas.imageMode(CENTER);
      for (int k = 0; k < LOGOS_PER_BOX; k++) {
        // where it goes round: part way out to the rectangle's edge, drifting in and out
        float orbit = 0.15 + 0.85 * rnd.nextFloat();
        float drift = 0.2 + 0.3 * rnd.nextFloat();
        float phase = TWO_PI * rnd.nextFloat();
        float speed = direction * TWO_PI / 60 * lerp(SWIRL_MIN, SWIRL_MAX, rnd.nextFloat());
        float spin = (rnd.nextBoolean() ? 1 : -1) * (0.2 + 0.6 * rnd.nextFloat());
        float shade = rnd.nextFloat();
        float scale = lerp(GRID_SCALE_MIN, GRID_SCALE_MAX, rnd.nextFloat());
        boolean broken = rnd.nextFloat() < BROKEN_CHANCE;
        long seed = rnd.nextLong();

        float angle = phase + speed * t;
        float r = orbit * (0.8 + 0.2 * sin(phase + drift * t));
        drawLogo((x0 + x1) / 2 + cos(angle) * r * w / 2, (y0 + y1) / 2 + sin(angle) * r * h / 2,
          logoW * scale, phase + spin * t, mix(pc, qc, shade), LOGO_ALPHA * fade,
          broken, seed, t, GRID_SCATTER, GRID_PART_SPIN);
      }
      canvas.noClip();
      canvas.imageMode(CORNER);
    }
  }
  canvas.noTint();
}

// one copy dropped on the floor
class Stamp {
  float x, y, size, angle;  // canvas pixels, and radians
  color colour;
  int born;                 // millis()
  boolean broken;
  long seed;                // its parts' paces, if broken
  // its own pace and starting point for circling and turning, so they don't move in step
  float swirlR, swirlPhase, swirlSpeed, spin;

  Stamp(float x, float y, color colour) {
    this.x = x;
    this.y = y;
    this.colour = colour;
    born = millis();
    size = lerp(TRAIL_SIZE_MIN_M, TRAIL_SIZE_MAX_M, sq(random(1))) * PIXELS_PER_METRE;
    angle = random(TWO_PI);
    broken = random(1) < BROKEN_CHANCE;
    seed = (long) random(Integer.MAX_VALUE);
    swirlR = TRAIL_SWIRL_M * PIXELS_PER_METRE * random(0.3, 1);
    swirlPhase = random(TWO_PI);
    // radians a second: about one turn per TRAIL_SWIRL_S, either way round
    swirlSpeed = (random(1) < 0.5 ? 1 : -1) * TWO_PI / TRAIL_SWIRL_S * random(0.6, 1.4);
    spin = radians(TRAIL_SPIN_DEG) * random(-1, 1);
  }
}

// drop a copy near anyone who has walked TRAIL_SPACING_M since their last
// one, and clear away the faded ones
void updateTrail() {
  for (Person p : people) {
    if (p.isLost()) continue;
    PVector here = new PVector(p.x, p.y);
    PVector last = lastStamp.get(p);
    if (last == null) {
      lastStamp.put(p, here);  // just arrived: the trail starts once they walk
    } else if (PVector.dist(here, last) >= TRAIL_SPACING_M) {
      PVector off = PVector.random2D().mult(random(TRAIL_SCATTER_M) * PIXELS_PER_METRE);
      trail.add(new Stamp(p.canvasX() + off.x, p.canvasY() + off.y, colourOf(p)));
      lastStamp.put(p, here);
    }
  }
  int now = millis();
  while (trail.size() > TRAIL_MAX || (trail.size() > 0 && now - trail.get(0).born > TRAIL_LIFE_S * 1000)) {
    trail.remove(0);
  }
}

void drawTrail() {
  float t = millis() / 1000.0;
  canvas.imageMode(CENTER);
  for (Stamp s : trail) {
    float age = (millis() - s.born) / 1000.0;
    // in over half a second, then out over the rest of its life: slowly at
    // first, faster towards the end, so most of the trail is clearly visible
    float alpha = TRAIL_ALPHA * min(1, age / 0.5) * (1 - sq(age / TRAIL_LIFE_S));
    if (alpha <= 0) continue;
    float a = s.swirlPhase + s.swirlSpeed * t;
    drawLogo(s.x + s.swirlR * cos(a), s.y + s.swirlR * sin(a), s.size, s.angle + s.spin * t,
      s.colour, alpha, s.broken, s.seed, t, TRAIL_SCATTER, TRAIL_PART_SPIN);
  }
  canvas.noTint();
  canvas.imageMode(CORNER);
}

// one copy on the FIELD floor
class FieldLogo {
  float homeX, homeY;  // canvas pixels: its spot
  float pushX, pushY;  // how far people have pushed it from there
  float size, angle;
  color colour;
  boolean broken;
  long seed;           // its parts' paces, if broken
  float swirlR, swirlPhase, swirlSpeed, spin;

  FieldLogo(float x, float y) {
    homeX = x;
    homeY = y;
    size = lerp(FIELD_SIZE_MIN_M, FIELD_SIZE_MAX_M, sq(random(1))) * PIXELS_PER_METRE;
    angle = random(TWO_PI);
    // a shade between two of the palette's colours
    colour = mix(PALETTE[(int) random(PALETTE.length)], PALETTE[(int) random(PALETTE.length)], random(1));
    broken = random(1) < BROKEN_CHANCE;
    seed = (long) random(Integer.MAX_VALUE);
    swirlR = TRAIL_SWIRL_M * PIXELS_PER_METRE * random(0.3, 1);
    swirlPhase = random(TWO_PI);
    swirlSpeed = (random(1) < 0.5 ? 1 : -1) * TWO_PI / TRAIL_SWIRL_S * random(0.6, 1.4);
    spin = radians(TRAIL_SPIN_DEG) * random(-1, 1);
  }
}

// cover the floor: one copy in each cell of a grid, somewhere in its cell, so
// they're spread evenly with no big gaps or clumps
void makeField() {
  field.clear();
  float area = canvas.width * canvas.height / sq(PIXELS_PER_METRE);
  float cell = sqrt(canvas.width * canvas.height / (area * FIELD_PER_M2));
  for (float y = 0; y < canvas.height; y += cell) {
    for (float x = 0; x < canvas.width; x += cell) {
      field.add(new FieldLogo(x + random(cell), y + random(cell)));
    }
  }
  lastFieldMillis = -1;
}

// push copies away from everyone's feet, and let the rest drift home
void updateField() {
  int now = millis();
  // seconds since last frame, capped so a stall doesn't make everything jump
  float dt = lastFieldMillis < 0 ? 0 : min(now - lastFieldMillis, 100) / 1000.0;
  lastFieldMillis = now;
  float settle = exp(-3 * dt / FIELD_RETURN_S);  // e^-3: 5% of the way left after FIELD_RETURN_S
  float pushStep = min(1, FIELD_PUSH_RATE * dt);
  for (FieldLogo f : field) {
    f.pushX *= settle;
    f.pushY *= settle;
    for (Person p : people) {
      if (p.isLost()) continue;
      float clear = FIELD_CLEAR_M * PIXELS_PER_METRE + 0.4 * f.size;
      float dx = f.homeX + f.pushX - p.canvasX(), dy = f.homeY + f.pushY - p.canvasY();
      float d = sqrt(dx * dx + dy * dy);
      if (d >= clear) continue;
      if (d < 1) {  // right underfoot: out whichever way it was swirling
        dx = cos(f.swirlPhase);
        dy = sin(f.swirlPhase);
        d = 1;
      }
      // part of the way out to `clear`, straight away from them
      float out = (clear - d) * pushStep;
      f.pushX += dx / d * out;
      f.pushY += dy / d * out;
    }
  }
}

void drawField() {
  float t = millis() / 1000.0;
  float clear = FIELD_CLEAR_M * PIXELS_PER_METRE;
  canvas.imageMode(CENTER);
  for (FieldLogo f : field) {
    float a = f.swirlPhase + f.swirlSpeed * t;
    // pushed further, broken apart further
    float pushed = min(1, sqrt(f.pushX * f.pushX + f.pushY * f.pushY) / clear);
    drawLogo(f.homeX + f.pushX + f.swirlR * cos(a), f.homeY + f.pushY + f.swirlR * sin(a), f.size,
      f.angle + f.spin * t, f.colour, FIELD_ALPHA, f.broken, f.seed, t,
      FIELD_SCATTER + FIELD_PUSH_SCATTER * pushed, FIELD_PART_SPIN);
  }
  canvas.noTint();
  canvas.imageMode(CORNER);
}

void keyPressed() {
  if (key == 'm' || key == 'M') {
    mode = (mode + 1) % MODE_NAMES.length;
    trail.clear();
    lastStamp.clear();
    if (mode == FIELD) makeField();  // a fresh floor, nothing pushed
    println(MODE_NAMES[mode] + " mode");
  }
}

// a colour part way (0-1) from a to b, mixed as light (see BLEND_AS_LIGHT)
color mix(color a, color b, float f) {
  return color(unlight(lerp(light(red(a)), light(red(b)), f)),
    unlight(lerp(light(green(a)), light(green(b)), f)),
    unlight(lerp(light(blue(a)), light(blue(b)), f)));
}

// a colour channel (0-255) as an amount of light, 0-1, and back
// (squared: close to the screen's gamma of 2.2, and cheap)
float light(float v) {
  return BLEND_AS_LIGHT ? sq(v / 255) : v / 255;
}
float unlight(float v) {
  return 255 * (BLEND_AS_LIGHT ? sqrt(v) : v);
}

color colourOf(Person p) {
  Integer c = colours.get(p);
  return c == null ? color(255) : c;
}

// someone new has stepped onto the floor (People tab): give them the colour
// fewest people have now, picked at random among equals
void personArrived(Person p) {
  int[] count = new int[PALETTE.length];
  for (int c : colours.values()) {
    for (int i = 0; i < PALETTE.length; i++) {
      if (PALETTE[i] == c) count[i]++;
    }
  }
  int fewest = min(count);
  IntList choices = new IntList();
  for (int i = 0; i < PALETTE.length; i++) {
    if (count[i] == fewest) choices.append(i);
  }
  colours.put(p, PALETTE[choices.get((int) random(choices.size()))]);
}

// someone has gone: lost, faded out and not found again within REATTACH_MS
// (People tab). While lost they were still in `people`, with isLost() true
void personLeft(Person p) {
  colours.remove(p);
  lastStamp.remove(p);  // their trail stays, fading
}
