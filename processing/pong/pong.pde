// pong - pong played the length of the floor between two people, with the
// Mill logo as the ball.
//
// Whoever stands in the left half plays from the left end, whoever stands in
// the right half from the right. Each player's bat is a bar at their feet, so
// they move it by walking: across the floor to reach the ball, or forward to
// meet it early (but not past BAT_MAX_REACH_M from their end). The ball comes
// off a bat at an angle set by where it hits, a little faster each time, and
// bounces off the long sides. Let it past your end and the logo bursts and
// the other player scores; first to POINTS_TO_WIN wins, and the game starts again.
//
// An end with nobody playing is a wall, so one person can play alone; with
// nobody on the floor the ball rallies wall to wall and the score is cleared.
// The score is a row of pips each side of the centre line (no digits, so it
// reads the same from both ends).
//
// Run the tracker alongside it:
//   python track_osc.py               # people, from the camera
//   python track_osc.py --simulate 2  # two imaginary people, no camera
// or hold the mouse button down for a test person at the mouse.
// Press r to clear the score.
//
// People tab: who is on the floor (copied unchanged from the template).
// Logo tab: the logo as tinted images, whole or broken into its parts.
// Projection tab: draw into `canvas` (PIXELS_PER_METRE pixels per metre,
// origin at the first tape mark), warped onto the floor.

// the logo (in data/; Logo tab), its lines thickened to LOGO_LINE_M metres
// on a LOGO_IMAGE_M-wide copy so they show on the floor
final String LOGO_FILE = "mill_only_logo.svg";
final float LOGO_LINE_M = 0.015;

// the ball: the logo this wide, bouncing as a circle of BALL_RADIUS_M
final float BALL_M = 0.26;
final float BALL_RADIUS_M = 0.07;
// metres a second: when served, the most it gets to, and how much faster each hit makes it
final float SERVE_SPEED = 0.8;
final float MAX_SPEED = 2.4;
final float SPEEDUP = 1.07;
// it leaves a bat at up to this angle from straight down the floor, by how far off-centre it hit
final float MAX_BOUNCE_ANGLE = radians(55);
// served at up to this angle
final float MAX_SERVE_ANGLE = radians(25);
// it spins this many radians per metre travelled
final float SPIN_PER_M = 1.5;

// the bats: this long, this thick, and no further than this from their end
final float BAT_M = 0.36;
final float BAT_THICK_M = 0.04;
final float BAT_MAX_REACH_M = 0.8;

final int POINTS_TO_WIN = 5;
// after a miss, the logo bursts for this long; then a pause before the next serve
final int SHATTER_MS = 1000;
// it bursts into this many small copies, flying this far
final int SHATTER_PIECES = 7;
final float SHATTER_SPREAD_M = 0.5;
final int SERVE_PAUSE_MS = 700;
// after a win, the winner's pips flash this long before the next game
final int WIN_MS = 3500;

// the players' colours (left, right), and the walls'
final color[] SIDE_COLOURS = {color(150, 255, 0), color(0, 215, 255)};
final color WALL_COLOUR = color(90);

final int LEFT_END = 0, RIGHT_END = 1;

// what the ball is doing
final int SERVING = 0, PLAYING = 1, SHATTERING = 2;
int state = SERVING;
int stateSince;  // millis() when it started doing it

float ballX, ballY, ballVX, ballVY;  // metres, metres a second
float ballSpeed;
float ballAngle;
// whose colour the ball has: the last to hit it, or -1 (white)
int lastHit = -1;
// the side the next serve goes towards
int serveTo = LEFT_END;
long shatterSeed;

Person[] players = new Person[2];
// each bat's x last frame, to catch a bat stepping into the ball
float[] lastBatX = new float[2];
int[] score = new int[2];
// the side that has just won, while the win shows, or -1
int winner = -1;
int wonAt;
int lastFrame = -1;

void settings() {
  // full screen on the projector, or a window when it isn't connected (Projection tab)
  chooseScreen();
}

void setup() {
  frameRate(60);
  setupProjection();
  loadLogo();
  startPeopleReceiver();
  serveTo = random(1) < 0.5 ? LEFT_END : RIGHT_END;
  startServe();
}

void draw() {
  updatePeople();
  int now = millis();
  float dt = lastFrame < 0 ? 0 : min(now - lastFrame, 50) / 1000.0;
  lastFrame = now;

  choosePlayers();
  if (players[LEFT_END] == null && players[RIGHT_END] == null) {
    score[LEFT_END] = score[RIGHT_END] = 0;
    winner = -1;
  }
  if (winner >= 0 && now - wonAt > WIN_MS) {
    winner = -1;
    score[LEFT_END] = score[RIGHT_END] = 0;
  }
  updateBall(dt, now);

  canvas.beginDraw();
  canvas.background(0);
  drawCourt(now);
  for (int side = LEFT_END; side <= RIGHT_END; side++) drawBat(side);
  drawBall(now);
  canvas.endDraw();

  // warp the canvas onto the taped area (Projection tab)
  drawProjected(canvas);
}

// --- players ---

// Keep each side's player while they're on the floor and in their half;
// otherwise the person in that half nearest its end takes over.
void choosePlayers() {
  for (int side = LEFT_END; side <= RIGHT_END; side++) {
    Person p = players[side];
    if (p != null && (!people.contains(p) || sideOf(p) != side)) p = null;
    if (p == null) {
      for (Person q : people) {
        if (q.isLost() || sideOf(q) != side || q == players[1 - side]) continue;
        if (p == null || distFromEnd(q, side) < distFromEnd(p, side)) p = q;
      }
      if (p != null) lastBatX[side] = batX(side, p);
    }
    players[side] = p;
  }
}

int sideOf(Person p) {
  return p.x < floorW / 2 ? LEFT_END : RIGHT_END;
}

float distFromEnd(Person p, int side) {
  return side == LEFT_END ? p.x : floorW - p.x;
}

// where a side's bat is: at its player's feet, but no nearer the middle than BAT_MAX_REACH_M
float batX(int side, Person p) {
  return side == LEFT_END ? min(p.x, BAT_MAX_REACH_M) : max(p.x, floorW - BAT_MAX_REACH_M);
}

float batY(Person p) {
  return constrain(p.y, 0, floorH);
}

// a lost player's bat fades with them, and stops hitting once mostly gone
boolean batSolid(Person p) {
  return p != null && p.shown > 0.3;
}

// --- the ball ---

void startServe() {
  state = SERVING;
  stateSince = millis();
  ballX = floorW / 2;
  ballY = floorH / 2;
  ballVX = ballVY = 0;
  lastHit = -1;
}

void updateBall(float dt, int now) {
  float[] batWas = lastBatX.clone();
  for (int side = LEFT_END; side <= RIGHT_END; side++) {
    if (players[side] != null) lastBatX[side] = batX(side, players[side]);
  }

  if (state == SERVING) {
    if (now - stateSince > SERVE_PAUSE_MS && winner < 0) {
      float a = random(-MAX_SERVE_ANGLE, MAX_SERVE_ANGLE);
      ballSpeed = SERVE_SPEED;
      ballVX = (serveTo == LEFT_END ? -1 : 1) * ballSpeed * cos(a);
      ballVY = ballSpeed * sin(a);
      state = PLAYING;
      stateSince = now;
    }
    return;
  }

  float lastX = ballX;
  ballX += ballVX * dt;
  ballY += ballVY * dt;
  ballAngle += SPIN_PER_M * ballSpeed * dt * (ballVX < 0 ? -1 : 1);

  if (state == SHATTERING) {
    // the pieces drift on, slowing
    ballVX *= pow(0.2, dt);
    ballVY *= pow(0.2, dt);
    if (now - stateSince > SHATTER_MS) startServe();
    return;
  }

  // the long sides
  if (ballY < BALL_RADIUS_M) {
    ballY = 2 * BALL_RADIUS_M - ballY;
    ballVY = abs(ballVY);
  } else if (ballY > floorH - BALL_RADIUS_M) {
    ballY = 2 * (floorH - BALL_RADIUS_M) - ballY;
    ballVY = -abs(ballVY);
  }

  for (int side = LEFT_END; side <= RIGHT_END; side++) {
    int dir = side == LEFT_END ? 1 : -1;  // the way the ball goes when this side hits it
    Person p = players[side];
    if (p != null) {
      // a hit: the ball's near edge has crossed the bat's line since last
      // frame (whichever of them moved), alongside the bat
      float bx = batX(side, p), by = batY(p);
      float before = (lastX - batWas[side]) * dir, after = (ballX - bx) * dir;
      float offset = (ballY - by) / (BAT_M / 2 + BALL_RADIUS_M);
      if (batSolid(p) && before >= BALL_RADIUS_M && after < BALL_RADIUS_M && abs(offset) <= 1) {
        ballSpeed = min(ballSpeed * SPEEDUP, MAX_SPEED);
        float a = offset * MAX_BOUNCE_ANGLE;
        ballVX = dir * ballSpeed * cos(a);
        ballVY = ballSpeed * sin(a);
        ballX = bx + dir * BALL_RADIUS_M;
        lastHit = side;
      }
    }

    // its end: a wall if nobody's playing it, otherwise a miss
    float end = side == LEFT_END ? 0 : floorW;
    if (p == null) {
      if ((ballX - end) * dir < BALL_RADIUS_M) {
        ballX = end + dir * BALL_RADIUS_M;
        ballVX = dir * abs(ballVX);
      }
    } else if ((ballX - end) * dir < 0) {
      missed(side, now);
      return;
    }
  }
}

// the ball got past this side
void missed(int side, int now) {
  state = SHATTERING;
  stateSince = now;
  shatterSeed = now;
  serveTo = side;
  if (winner < 0 && players[1 - side] != null) {
    score[1 - side]++;
    if (score[1 - side] >= POINTS_TO_WIN) {
      winner = 1 - side;
      wonAt = now;
    }
  }
}

// --- drawing ---

void drawCourt(int now) {
  float ppm = PIXELS_PER_METRE;

  // the dashed centre line
  canvas.stroke(70);
  canvas.strokeWeight(0.015 * ppm);
  int dashes = 9;
  for (int i = 0; i < dashes; i++) {
    float y0 = floorH * (i + 0.25) / dashes, y1 = floorH * (i + 0.75) / dashes;
    canvas.line(floorW / 2 * ppm, y0 * ppm, floorW / 2 * ppm, y1 * ppm);
  }

  // an end with nobody playing it is a wall
  canvas.stroke(WALL_COLOUR);
  canvas.strokeWeight(0.03 * ppm);
  if (players[LEFT_END] == null) canvas.line(0.015 * ppm, 0, 0.015 * ppm, canvas.height);
  if (players[RIGHT_END] == null) canvas.line(canvas.width - 0.015 * ppm, 0, canvas.width - 0.015 * ppm, canvas.height);

  // the score: a column of pips either side of the centre line, filled as points are won
  float pip = 0.06 * ppm;
  for (int side = LEFT_END; side <= RIGHT_END; side++) {
    if (players[side] == null && score[side] == 0) continue;
    float x = (floorW / 2 + (side == LEFT_END ? -0.15 : 0.15)) * ppm;
    boolean flashOff = winner == side && (now - wonAt) / 250 % 2 == 1;
    for (int i = 0; i < POINTS_TO_WIN; i++) {
      float y = (floorH / 2 + (i - (POINTS_TO_WIN - 1) / 2.0) * 0.12) * ppm;
      canvas.stroke(SIDE_COLOURS[side], 160);
      canvas.strokeWeight(0.008 * ppm);
      if (i < score[side] && !flashOff) {
        canvas.fill(SIDE_COLOURS[side]);
      } else {
        canvas.noFill();
      }
      canvas.ellipse(x, y, pip, pip);
    }
  }
}

void drawBat(int side) {
  Person p = players[side];
  if (p == null || p.shown == 0) return;
  float ppm = PIXELS_PER_METRE;
  canvas.noStroke();
  canvas.fill(SIDE_COLOURS[side], 255 * p.shown);
  canvas.rectMode(CENTER);
  canvas.rect(batX(side, p) * ppm, batY(p) * ppm, BAT_THICK_M * ppm, BAT_M * ppm, BAT_THICK_M * ppm / 2);
  canvas.rectMode(CORNER);
}

void drawBall(int now) {
  float ppm = PIXELS_PER_METRE;
  color col = lastHit < 0 ? color(255) : SIDE_COLOURS[lastHit];
  canvas.imageMode(CENTER);
  if (state == SHATTERING) {
    // bursts into small copies of itself, flying apart, spinning and fading
    float f = (now - stateSince) / (float) SHATTER_MS;
    for (int k = 0; k < SHATTER_PIECES; k++) {
      float a = TWO_PI * (k + hash01(shatterSeed, 3 * k)) / SHATTER_PIECES;
      float r = SHATTER_SPREAD_M * lerp(0.6, 1, hash01(shatterSeed, 3 * k + 1)) * sqrt(f);
      float spin = 8 * (2 * hash01(shatterSeed, 3 * k + 2) - 1);
      drawLogo((ballX + r * cos(a)) * ppm, (ballY + r * sin(a)) * ppm, BALL_M * ppm * lerp(0.5, 0.25, f),
        ballAngle + spin * f, col, 255 * (1 - f), false, 0, 0, 0, 0);
    }
  } else {
    // fades in at the centre while waiting to be served
    float alpha = state == SERVING ? 255 * min(1, (now - stateSince) / (float) SERVE_PAUSE_MS) : 255;
    if (winner >= 0) alpha = 0;
    drawLogo(ballX * ppm, ballY * ppm, BALL_M * ppm, ballAngle, col, alpha, false, 0, 0, 0, 0);
  }
  canvas.noTint();
  canvas.imageMode(CORNER);
}

void keyPressed() {
  if (key == 'r') {
    score[LEFT_END] = score[RIGHT_END] = 0;
    winner = -1;
    startServe();
  }
}

// someone new has stepped onto the floor (People tab)
void personArrived(Person p) {
  println("arrived: #" + p.id + " at " + nf(p.x, 0, 2) + ", " + nf(p.y, 0, 2) + " m");
}

// someone has gone: lost, faded out and not found again within REMEMBER_MS (People tab)
void personLeft(Person p) {
  println("left: #" + p.id);
}
