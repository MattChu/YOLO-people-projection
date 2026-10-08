// Projection tab: the piece is drawn into `canvas`, which stands for the taped
// rectangle on the floor (PIXELS_PER_METRE pixels per metre, origin at the
// first tape mark - the same coordinates track_osc.py sends). drawProjected()
// warps it onto the projector with the floor -> projector mapping measured by
// calibrate.py, so the piece fills the tape exactly and circles land round.
//
// The projector is found by matching the screen position saved in the
// calibration. If it isn't connected, the sketch opens a half-size window
// showing the same view, for testing. Without a calibration file, the canvas
// simply fills that window.
//
// The calibration is calibrate.py's calibration.json, found in this sketch's
// folder or the nearest folder above it, so a sketch works anywhere inside
// the project folder (processing/, or my-sketches/ for your own).

import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;

final String CALIBRATION = "calibration.json";
final String PROJECTOR = "proj0";
final float PIXELS_PER_METRE = 500;
// grid cells along each side of the warp: exact at the grid points, smooth between
final int MESH = 24;

PGraphics canvas;
JSONObject projector;          // this projector's calibration, or null
float floorW = 2.4, floorH = 1.4;  // taped rectangle in metres, replaced from the calibration
int projW = 1280, projH = 800;
float[][] floorToScreen;       // 3x3: floor metres -> this window's pixels
float[][] floorFromScreen;     // the inverse
PVector[][] mesh;              // floor grid points, in window pixels

// CALIBRATION in this sketch's folder or the nearest one above it, or null
File findCalibration() {
  for (File dir = new File(sketchPath()); dir != null; dir = dir.getParentFile()) {
    File file = new File(dir, CALIBRATION);
    if (file.exists()) return file;
  }
  return null;
}

// called from settings(): full screen on the projector if it's connected
void chooseScreen() {
  File file = findCalibration();
  if (file != null) {
    println("Calibration: " + file);
    JSONObject cal = loadJSONObject(file);
    JSONObject floor = cal.getJSONObject("floor");
    floorW = floor.getFloat("width_m");
    floorH = floor.getFloat("height_m");
    if (cal.getJSONObject("projectors").hasKey(PROJECTOR)) {
      projector = cal.getJSONObject("projectors").getJSONObject(PROJECTOR);
      projW = projector.getJSONArray("resolution").getInt(0);
      projH = projector.getJSONArray("resolution").getInt(1);
    }
  } else {
    println("No " + CALIBRATION + " in " + sketchPath() + " or above it: the canvas just fills the window");
  }

  int display = projector == null ? 0 : displayAt(projector.getJSONArray("position").getInt(0),
    projector.getJSONArray("position").getInt(1), projW, projH);
  if (display > 0) {
    fullScreen(P2D, display);
  } else {
    if (projector != null) println("Projector not found at its calibrated position: showing a test window");
    size(projW / 2, projH / 2, P2D);
  }
}

// Processing's display number (1, 2...) of the screen at this desktop position and size, or 0
int displayAt(int x, int y, int w, int h) {
  GraphicsDevice[] screens = GraphicsEnvironment.getLocalGraphicsEnvironment().getScreenDevices();
  for (int i = 0; i < screens.length; i++) {
    Rectangle r = screens[i].getDefaultConfiguration().getBounds();
    if (r.x == x && r.y == y && r.width == w && r.height == h) return i + 1;
  }
  return 0;
}

// called from setup(), once the window exists
void setupProjection() {
  canvas = createGraphics(round(floorW * PIXELS_PER_METRE), round(floorH * PIXELS_PER_METRE), P2D);
  if (projector != null) {
    // calibrated: floor -> projector pixels, scaled to this window (smaller when testing)
    float[][] H = new float[3][3];
    JSONArray rows = projector.getJSONArray("H_floor_to_proj");
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) H[r][c] = rows.getJSONArray(r).getFloat(c);
    }
    float sx = width / (float) projW, sy = height / (float) projH;
    for (int c = 0; c < 3; c++) {
      H[0][c] *= sx;
      H[1][c] *= sy;
    }
    floorToScreen = H;
  } else {
    floorToScreen = new float[][] {{width / floorW, 0, 0}, {0, height / floorH, 0}, {0, 0, 1}};
  }
  floorFromScreen = invert3(floorToScreen);

  mesh = new PVector[MESH + 1][MESH + 1];
  for (int i = 0; i <= MESH; i++) {
    for (int j = 0; j <= MESH; j++) {
      mesh[i][j] = applyH(floorToScreen, floorW * i / MESH, floorH * j / MESH);
    }
  }
}

// draw the canvas onto the floor; everything outside the taped area stays black
void drawProjected(PGraphics c) {
  background(0);
  noStroke();
  textureMode(NORMAL);  // texture coordinates 0..1 across the canvas
  beginShape(QUADS);
  texture(c);
  for (int i = 0; i < MESH; i++) {
    for (int j = 0; j < MESH; j++) {
      float u0 = i / (float) MESH, u1 = (i + 1) / (float) MESH;
      float v0 = j / (float) MESH, v1 = (j + 1) / (float) MESH;
      vertex(mesh[i][j].x, mesh[i][j].y, u0, v0);
      vertex(mesh[i + 1][j].x, mesh[i + 1][j].y, u1, v0);
      vertex(mesh[i + 1][j + 1].x, mesh[i + 1][j + 1].y, u1, v1);
      vertex(mesh[i][j + 1].x, mesh[i][j + 1].y, u0, v1);
    }
  }
  endShape();
}

// a window position (e.g. the mouse) on the floor, in metres
PVector screenToFloor(float x, float y) {
  return applyH(floorFromScreen, x, y);
}

PVector applyH(float[][] H, float x, float y) {
  float w = H[2][0] * x + H[2][1] * y + H[2][2];
  return new PVector((H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w);
}

float[][] invert3(float[][] m) {
  float a = m[0][0], b = m[0][1], c = m[0][2];
  float d = m[1][0], e = m[1][1], f = m[1][2];
  float g = m[2][0], h = m[2][1], k = m[2][2];
  float det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  return new float[][] {
    {(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det},
    {(f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det},
    {(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det}};
}
