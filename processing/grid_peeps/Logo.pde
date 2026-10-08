// Logo tab: the logo (LOGO_FILE, in data/), whole or broken into its parts,
// as white images that are tinted as they're drawn: far quicker than drawing
// the vector shape hundreds of times a frame.
//
// The parts come from the SVG itself. Its path is a set of outlines; one
// inside another is a hole in it (or something drawn in that hole), so each
// part is an outline with everything inside it. For the Mill logo that's the
// gear-and-house (one joined outline in the file, so they stay together) and
// each letter of MILL.

// the images are drawn for a logo this wide, at twice the canvas resolution;
// drawn bigger or smaller, their lines scale with them
final float LOGO_IMAGE_M = 0.4;

LogoPart logoWhole;
LogoPart[] logoParts;

class LogoPart {
  PImage img;
  // its centre, relative to the logo's centre, and its size (image included
  // its padding), all as fractions of the logo's width
  float cx, cy, w, h;
}

void loadLogo() {
  XML svg = loadXML(LOGO_FILE);
  float[] viewBox = float(splitTokens(svg.getString("viewBox"), " ,"));
  float vbW = viewBox[2], vbH = viewBox[3];
  XML path = svg.getChild("path");
  String d = path.getString("d");

  // the outlines, each starting with a move (M), and the box round each
  String[] outlines = splitBefore(d, 'M');
  float[][] boxes = new float[outlines.length][];
  for (int i = 0; i < outlines.length; i++) boxes[i] = boundingBox(outlines[i]);

  logoWhole = renderPart(svg, path, d, new float[] {0, 0, vbW, vbH}, vbW, vbH);

  // a part for each outline not inside another, with the ones inside it
  ArrayList<LogoPart> parts = new ArrayList<LogoPart>();
  for (int i = 0; i < outlines.length; i++) {
    boolean inside = false;
    for (int j = 0; j < outlines.length; j++) {
      // (two the same size: the first one holds the other)
      if (j != i && contains(boxes[j], boxes[i]) && !(j > i && contains(boxes[i], boxes[j]))) inside = true;
    }
    if (inside) continue;
    String partD = "";
    for (int j = 0; j < outlines.length; j++) {
      if (j == i || (contains(boxes[i], boxes[j]) && !(j < i && contains(boxes[j], boxes[i])))) partD += outlines[j] + " ";
    }
    parts.add(renderPart(svg, path, partD, boxes[i], vbW, vbH));
  }
  logoParts = parts.toArray(new LogoPart[0]);
  println(LOGO_FILE + ": " + outlines.length + " outlines, " + logoParts.length + " parts");
}

// one part: the outlines in d (inside box, in the SVG's units) drawn white,
// filled and with their lines thickened to LOGO_LINE_M
LogoPart renderPart(XML svg, XML path, String d, float[] box, float vbW, float vbH) {
  path.setString("d", d);
  PShape shape = new PShapeSVG(svg);
  shape.disableStyle();

  float scale = LOGO_IMAGE_M * PIXELS_PER_METRE * 2 / vbW;  // image pixels per SVG unit
  float line = LOGO_LINE_M * PIXELS_PER_METRE * 2;
  float pad = line;  // room for the thickened lines
  PGraphics g = createGraphics(ceil((box[2] - box[0]) * scale + 2 * pad), ceil((box[3] - box[1]) * scale + 2 * pad), P2D);
  g.beginDraw();
  g.clear();
  g.fill(255);
  g.stroke(255);
  g.strokeWeight(line);
  g.strokeJoin(ROUND);
  g.shape(shape, pad - box[0] * scale, pad - box[1] * scale, vbW * scale, vbH * scale);
  g.endDraw();

  LogoPart part = new LogoPart();
  part.img = g.get();
  part.w = g.width / scale / vbW;
  part.h = g.height / scale / vbW;
  part.cx = (box[0] - pad / scale + g.width / scale / 2 - vbW / 2) / vbW;
  part.cy = (box[1] - pad / scale + g.height / scale / 2 - vbH / 2) / vbW;
  return part;
}

// One copy of the logo, centred at x, y (canvas pixels), `size` pixels wide,
// turned by angle, tinted col at alpha. A broken copy is drawn as its parts,
// each circling its own place in the logo, up to `scatter` of the size away,
// and turning on its own at up to partSpin radians a second; `seed` gives
// each copy's parts their own paces, t is the time in seconds.
// Call between canvas.imageMode(CENTER) and canvas.noTint().
void drawLogo(float x, float y, float size, float angle, color col, float alpha,
  boolean broken, long seed, float t, float scatter, float partSpin) {
  canvas.tint(col, alpha);
  canvas.pushMatrix();
  canvas.translate(x, y);
  canvas.rotate(angle);
  if (!broken) {
    canvas.image(logoWhole.img, logoWhole.cx * size, logoWhole.cy * size, logoWhole.w * size, logoWhole.h * size);
  } else {
    for (int k = 0; k < logoParts.length; k++) {
      LogoPart p = logoParts[k];
      float phase = TWO_PI * hash01(seed, 4 * k);
      float orbitSpeed = lerp(0.3, 1.0, hash01(seed, 4 * k + 1)) * (hash01(seed, 4 * k + 2) < 0.5 ? 1 : -1);
      float r = scatter * size * hash01(seed, 4 * k + 3);
      float a = phase + orbitSpeed * t;
      canvas.pushMatrix();
      canvas.translate(p.cx * size + r * cos(a), p.cy * size + r * sin(a));
      canvas.rotate(partSpin * (2 * hash01(seed, 4 * k + 2) - 1) * t + phase);
      canvas.image(p.img, 0, 0, p.w * size, p.h * size);
      canvas.popMatrix();
    }
  }
  canvas.popMatrix();
}

// a number from 0 to 1 that's always the same for the same seed and k
float hash01(long seed, int k) {
  long h = seed * 0x9E3779B97F4A7C15L + (k + 1) * 0xC2B2AE3D27D4EB4FL;
  h ^= h >>> 31;
  h *= 0xBF58476D1CE4E5B9L;
  h ^= h >>> 29;
  return (h >>> 40) / (float) (1L << 24);
}

// s cut up before each c (anything before the first c is dropped)
String[] splitBefore(String s, char c) {
  ArrayList<String> pieces = new ArrayList<String>();
  int start = s.indexOf(c);
  while (start >= 0) {
    int next = s.indexOf(c, start + 1);
    pieces.add(next < 0 ? s.substring(start) : s.substring(start, next));
    start = next;
  }
  return pieces.toArray(new String[0]);
}

// the box round an outline's points (curve handles included, so a little
// generous): {left, top, right, bottom}
float[] boundingBox(String outline) {
  float[] box = {Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
  String[][] points = matchAll(outline, "(-?[0-9.]+)\\s*,\\s*(-?[0-9.]+)");
  if (points == null) return new float[] {0, 0, 0, 0};
  for (String[] pt : points) {
    float x = float(pt[1]), y = float(pt[2]);
    box[0] = min(box[0], x);
    box[1] = min(box[1], y);
    box[2] = max(box[2], x);
    box[3] = max(box[3], y);
  }
  return box;
}

boolean contains(float[] outer, float[] inner) {
  return outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3];
}
