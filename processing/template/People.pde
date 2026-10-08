// People tab: who is on the floor, and where. The same in every sketch:
// copy it unchanged, and keep a sketch's own reactions to people in its main tab.
//
// track_osc.py sends /person id x y (x, y in metres from the first tape mark)
// for every tracked person, every frame, over OSC/UDP. A small receiver thread
// keeps the latest sighting per tracker id. Nothing is sent when someone
// leaves, so a tracker id not heard from for PERSON_TIMEOUT_MS is dropped
// (which also covers lost packets).
//
// Call updatePeople() once at the start of draw(). It keeps `people`, one
// Person each, steadier than the tracker's ids:
// - a Person glides between sightings (which arrive as often as the tracker
//   runs, often less often than the sketch draws) and fades in when they
//   arrive (`shown`, 0-1)
// - when the tracker loses them, they stay in `people`, lost (isLost()), and
//   fade out. The tracker sometimes loses someone for a moment and finds them
//   again under a new id: if a new id turns up within REATTACH_DIST_M of a
//   lost Person within REATTACH_MS, it becomes that Person again
// - personArrived(p) and personLeft(p), in the main tab, are called when a
//   Person is made and when a lost one is finally forgotten
//
// Hold the mouse button down for a test person at the mouse.

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.nio.ByteBuffer;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

final int OSC_PORT = 12000;
// drop a tracker id not heard from for this long: its Person is lost
final int PERSON_TIMEOUT_MS = 500;
// fraction of the way a Person moves towards their latest sighting per frame at 60 fps
final float EASE = 0.25;
// people fade in when they arrive and out when they're lost, over this long
final int FADE_MS = 400;
// a lost Person is kept this long...
final int REATTACH_MS = 2000;
// ...for a new tracker id first seen within this many metres of them to take over
final float REATTACH_DIST_M = 0.5;
// the mouse's test person
final int MOUSE_ID = -1;

class Person {
  // the tracker's id when they were first seen: stays the same when it renumbers them
  int id;
  // the tracker's id for them now
  int trackId;
  // where they are, in metres, gliding towards tx, ty: where they were last seen
  float x, y, tx, ty;
  // how visible to make them, 0-1: fades in when they arrive and out when they're lost
  float shown = 0;
  // millis() when the tracker lost them, or -1 while it has them
  int lostAt = -1;

  Person(int id, float x, float y) {
    this.id = this.trackId = id;
    this.x = this.tx = x;
    this.y = this.ty = y;
  }
  boolean isLost() {
    return lostAt >= 0;
  }
  // where they are on the canvas, in pixels
  float canvasX() {
    return x * PIXELS_PER_METRE;
  }
  float canvasY() {
    return y * PIXELS_PER_METRE;
  }
}

// the tracker's latest word on one of its ids
class Sighting {
  float x, y;  // metres
  int heard;   // millis() when heard
  Sighting(float x, float y, int heard) {
    this.x = x;
    this.y = y;
    this.heard = heard;
  }
}

// written by the receiver thread, read in draw(): needs to be thread-safe
ConcurrentHashMap<Integer, Sighting> sightings = new ConcurrentHashMap<Integer, Sighting>();
// everyone on the floor, and those lately lost (only touched in draw())
ArrayList<Person> people = new ArrayList<Person>();
int lastPeopleUpdate = -1;

void startPeopleReceiver() {
  Thread receiver = new Thread(new Runnable() {
    public void run() {
      receivePeople();
    }
  });
  receiver.setDaemon(true);  // don't keep the sketch running after it closes
  receiver.start();
}

void receivePeople() {
  DatagramSocket socket;
  try {
    socket = new DatagramSocket(OSC_PORT);
  }
  catch (Exception e) {
    println("Can't listen for people on port " + OSC_PORT + ": " + e.getMessage());
    return;
  }
  println("Listening for /person on port " + OSC_PORT);
  byte[] buffer = new byte[1024];
  while (true) {
    try {
      DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
      socket.receive(packet);
      readPerson(ByteBuffer.wrap(packet.getData(), 0, packet.getLength()));
    }
    catch (Exception e) {
      // a malformed packet: ignore it and carry on listening
    }
  }
}

// An OSC message is its address and type tags, each a string ended by a zero
// byte and padded to a multiple of 4 bytes, then the arguments: here an int
// and two floats, big-endian (ByteBuffer's default).
void readPerson(ByteBuffer b) {
  if (!readOscString(b).equals("/person")) return;
  if (!readOscString(b).equals(",iff")) return;
  int id = b.getInt();
  float x = b.getFloat();
  float y = b.getFloat();
  sightings.put(id, new Sighting(x, y, millis()));
}

String readOscString(ByteBuffer b) {
  int start = b.position();
  while (b.get() != 0) {
  }
  String s = new String(b.array(), start, b.position() - 1 - start);
  b.position((b.position() + 3) / 4 * 4);
  return s;
}

void updatePeople() {
  int now = millis();
  // capped, so a long stall (e.g. dragging the window) doesn't make people jump
  float dt = lastPeopleUpdate < 0 ? 0 : min(now - lastPeopleUpdate, 100);
  lastPeopleUpdate = now;

  if (mousePressed) {
    PVector m = screenToFloor(mouseX, mouseY);
    sightings.put(MOUSE_ID, new Sighting(m.x, m.y, now));
  }

  // forget tracker ids not heard from lately (remove only if not updated meanwhile)
  for (Map.Entry<Integer, Sighting> e : sightings.entrySet()) {
    if (now - e.getValue().heard > PERSON_TIMEOUT_MS) sightings.remove(e.getKey(), e.getValue());
  }

  // follow everyone the tracker still has; the rest are lost, and forgotten
  // once lost too long. A lost Person whose id is heard again is back
  HashSet<Integer> claimed = new HashSet<Integer>();
  for (int i = people.size() - 1; i >= 0; i--) {
    Person p = people.get(i);
    Sighting s = sightings.get(p.trackId);
    if (s != null) {
      p.lostAt = -1;
      p.tx = s.x;
      p.ty = s.y;
      claimed.add(p.trackId);
    } else if (!p.isLost()) {
      p.lostAt = now;
    } else if (now - p.lostAt > REATTACH_MS) {
      people.remove(i);
      personLeft(p);
    }
  }

  // tracker ids no Person has yet
  ArrayList<Integer> newIds = new ArrayList<Integer>();
  for (int id : sightings.keySet()) {
    if (!claimed.contains(id)) newIds.add(id);
  }

  // a new id next to a lost Person becomes them, the closest pair first (so two
  // people lost and found together are each themselves again)
  while (true) {
    Person best = null;
    int bestId = 0;
    float bestDistSq = REATTACH_DIST_M * REATTACH_DIST_M;
    for (int id : newIds) {
      Sighting s = sightings.get(id);
      if (s == null) continue;
      for (Person p : people) {
        if (!p.isLost()) continue;
        float dx = s.x - p.tx, dy = s.y - p.ty;
        if (dx * dx + dy * dy <= bestDistSq) {
          best = p;
          bestId = id;
          bestDistSq = dx * dx + dy * dy;
        }
      }
    }
    if (best == null) break;
    best.trackId = bestId;
    best.lostAt = -1;
    best.tx = sightings.get(bestId).x;
    best.ty = sightings.get(bestId).y;
    newIds.remove((Integer) bestId);
  }

  // everyone else is new
  for (int id : newIds) {
    Sighting s = sightings.get(id);
    if (s == null) continue;
    Person p = new Person(id, s.x, s.y);
    people.add(p);
    personArrived(p);
  }

  // glide towards the latest sightings, and fade in or out
  float ease = 1 - pow(1 - EASE, dt / (1000.0 / 60));  // EASE per 60 fps frame
  float fade = dt / FADE_MS;
  for (Person p : people) {
    p.x += (p.tx - p.x) * ease;
    p.y += (p.ty - p.y) * ease;
    p.shown = p.isLost() ? max(0, p.shown - fade) : min(1, p.shown + fade);
  }
}

void mouseReleased() {
  sightings.remove(MOUSE_ID);
}
