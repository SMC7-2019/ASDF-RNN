import com.runwayml.*;

float SCORE_THRESHOLD = 0.6;

String BASEFOLDER = "";
String  EXTENSION = "png";
String      STYLE = "mixed";

String framesFolder=BASEFOLDER + "/frames";
String jsonFolder= BASEFOLDER + "/json";

int     FRAMERATE = 30;
int  CLIPDURATION = 8;
int FRAMESPERCLIP = FRAMERATE * CLIPDURATION;
int   UNAVAILABLE = 20; //If the clip has UNAVAILABLE or more blank frames in a row, skip it
float   SCALE_REF = 0.2;

int[][] connections = {
  {ModelUtils.POSE_RIGHT_EYE_INDEX, ModelUtils.POSE_LEFT_EYE_INDEX}, 

  {ModelUtils.POSE_RIGHT_SHOULDER_INDEX, ModelUtils.POSE_RIGHT_ELBOW_INDEX}, 
  {ModelUtils.POSE_RIGHT_ELBOW_INDEX, ModelUtils.POSE_RIGHT_WRIST_INDEX}, 
  {ModelUtils.POSE_LEFT_SHOULDER_INDEX, ModelUtils.POSE_LEFT_ELBOW_INDEX}, 
  {ModelUtils.POSE_LEFT_ELBOW_INDEX, ModelUtils.POSE_LEFT_WRIST_INDEX}, 
  {ModelUtils.POSE_RIGHT_HIP_INDEX, ModelUtils.POSE_RIGHT_KNEE_INDEX}, 
  {ModelUtils.POSE_RIGHT_KNEE_INDEX, ModelUtils.POSE_RIGHT_ANKLE_INDEX}, 
  {ModelUtils.POSE_LEFT_HIP_INDEX, ModelUtils.POSE_LEFT_KNEE_INDEX}, 
  {ModelUtils.POSE_LEFT_KNEE_INDEX, ModelUtils.POSE_LEFT_ANKLE_INDEX}, 

  {ModelUtils.POSE_RIGHT_SHOULDER_INDEX, ModelUtils.POSE_LEFT_SHOULDER_INDEX}, 
  {ModelUtils.POSE_LEFT_SHOULDER_INDEX, ModelUtils.POSE_LEFT_HIP_INDEX}, 
  {ModelUtils.POSE_LEFT_HIP_INDEX, ModelUtils.POSE_RIGHT_HIP_INDEX}, 
  {ModelUtils.POSE_RIGHT_HIP_INDEX, ModelUtils.POSE_RIGHT_SHOULDER_INDEX}, 

};

int[] head = {
  ModelUtils.POSE_RIGHT_EYE_INDEX, 
  ModelUtils.POSE_LEFT_EYE_INDEX, 
  ModelUtils.POSE_LEFT_EAR_INDEX, 
  ModelUtils.POSE_RIGHT_EAR_INDEX, 
  ModelUtils.POSE_NOSE_INDEX, 
};


RunwayHTTP runway;

PImage frame;
int currentFrame;
int currentClip;
int skippedCounter;
int unavailableCounter;
int lastRealFrame;

JSONObject fullData;
JSONArray framesData;

void setup() {

  size(600, 400);
  background(255);
  textSize(15);
  frameRate(60);

  currentClip=32;
  initBatch();

  runway = new RunwayHTTP(this);
  runway.setAutoUpdate(false);
}

void draw() {

  String filename = getFilename(FRAMESPERCLIP * currentClip + currentFrame);
  frame = loadImage(filename);

  if (frame == null) {
    noLoop();
    return;
  }

  //image(frame, 0, 0);
  sendFrameToRunway();

  if ((unavailableCounter >= UNAVAILABLE) || (currentFrame==0 && unavailableCounter>0)) {
    println("Skipping clip " + currentClip);
    initBatch();
    return;
  }

  save("stick/stick_"+ nf(currentClip, 3) + "_"+ nf(currentFrame, 8) + ".png");
  currentFrame++;

  if (currentFrame >= FRAMESPERCLIP) {

    if (unavailableCounter >0 ) {
      println("Skipping clip " + currentClip + " at end");
      initBatch();
      return;
    }

    initBatch();
  }

  
}

void sendFrameToRunway() {

  JSONObject input = new JSONObject();

  input.setString("image", ModelUtils.toBase64(frame));
  input.setString("estimationType", "Single Pose");
  input.setInt("maxPoseDetections", 1);
  input.setFloat("scoreThreshold", SCORE_THRESHOLD);

  runway.query(input.toString());
}

void drawParts(JSONObject data) {

  fill(255);
  noStroke();
  rect(0, 0, frame.width, frame.height);


  JSONArray head = data.getJSONArray("head");
  JSONArray keypoints = data.getJSONArray("data");

  pushMatrix();
  translate(0, 0);
  scale(frame.width, frame.height);  

  //WTF, Processing
  strokeWeight(0.003);

  stroke(0);
  for (int i = 0; i < connections.length; i++) {

    JSONArray startPart = keypoints.getJSONArray(connections[i][0]);
    JSONArray endPart   = keypoints.getJSONArray(connections[i][1]);
    float startX = startPart.getFloat(0);
    float startY = startPart.getFloat(1);
    float endX   = endPart.getFloat(0);
    float endY   = endPart.getFloat(1);
    line(startX, startY, endX, endY);
  }

  ellipse(head.getFloat(0), head.getFloat(1), 15.0/frame.width, 20.0/frame.height);

  popMatrix();
}




/*

 d8888b. db    db d8b   db db   d8b   db  .d8b.  db    db
 88  `8D 88    88 888o  88 88   I8I   88 d8' `8b `8b  d8'
 88oobY' 88    88 88V8o 88 88   I8I   88 88ooo88  `8bd8'
 88`8b   88    88 88 V8o88 Y8   I8I   88 88~~~88    88
 88 `88. 88b  d88 88  V888 `8b d8'8b d8' 88   88    88
 88   YD ~Y8888P' VP   V8P  `8b8' `8d8'  YP   YP    YP
 
 */

void runwayDataEvent(JSONObject runwayData) {

  JSONObject frameData = new JSONObject();

  if (runwayData.getJSONArray("scores").size() > 0) {

    JSONArray mainPose = runwayData.getJSONArray("poses").getJSONArray(0);

    //Translate the head
    JSONArray headCenter = getCenter(head, mainPose);
    JSONArray headCentered = new JSONArray();
    headCentered.setFloat(0, headCenter.getFloat(0));
    headCentered.setFloat(1, headCenter.getFloat(1));
    headCentered.setFloat(0, headCentered.getFloat(0));
    headCentered.setFloat(1, headCentered.getFloat(1));

    //Translate the joints
    JSONArray centeredPose = new JSONArray();
    for (int f=0; f<mainPose.size(); f++) {
      JSONArray joint = mainPose.getJSONArray(f);
      JSONArray jointCentered = new JSONArray();
      jointCentered.setFloat(0, joint.getFloat(0));
      jointCentered.setFloat(1, joint.getFloat(1) );
      jointCentered.setFloat(0, jointCentered.getFloat(0));
      jointCentered.setFloat(1, jointCentered.getFloat(1));
      centeredPose.setJSONArray(f, jointCentered);
    }

    frameData.setBoolean("interpolation", false);
    frameData.setFloat("score", runwayData.getJSONArray("scores").getFloat(0));
    frameData.setJSONArray("head", headCentered);
    frameData.setJSONArray("data", centeredPose);

    drawParts(frameData);

    if (unavailableCounter>0) {
    }
    unavailableCounter = 0;
    lastRealFrame = currentFrame;
  } else {
    //TODO: Interpolate features, torso and head
    frameData.setBoolean("interpolation", true);
    frameData.setFloat("score", 0);
    skippedCounter++;
    unavailableCounter++;
  }

  framesData.setJSONObject(currentFrame, frameData);
}

public void runwayInfoEvent(JSONObject info) {
  //println(info);
}

public void runwayErrorEvent(String message) {
  println(message);
}



/*

 db   db d88888b db      d8888b. d88888b d8888b. .d8888.
 88   88 88'     88      88  `8D 88'     88  `8D 88'  YP
 88ooo88 88ooooo 88      88oodD' 88ooooo 88oobY' `8bo.
 88~~~88 88~~~~~ 88      88~~~   88~~~~~ 88`8b     `Y8b.
 88   88 88.     88booo. 88      88.     88 `88. db   8D
 YP   YP Y88888P Y88888P 88      Y88888P 88   YD `8888Y'
 
 */

JSONArray getCenter(int[] markers, JSONArray data) {

  JSONArray center = new JSONArray();

  PVector vCenter = new PVector(0, 0);
  for (int i = 0; i < markers.length; i++) {
    JSONArray part = data.getJSONArray(markers[i]);
    vCenter.add(part.getFloat(0), part.getFloat(1));
  }  
  vCenter.div(markers.length);   

  center.setFloat(0, vCenter.x);
  center.setFloat(1, vCenter.y);

  return center;
}

public String getFilename(int index) {
  return BASEFOLDER + "frames_newbatch/frame_" + nf(index, 8) + "." + EXTENSION;
}

public void initBatch() {
  fullData = new JSONObject();
  framesData = new JSONArray();
  currentClip++;
  currentFrame=0;
  skippedCounter=0;
  unavailableCounter=0;
  lastRealFrame=0;
  println("Clip " + currentClip);
}

public JSONArray interpolateFeature (JSONArray startFeature, JSONArray endFeature, int currStep, int totalSteps) {

  PVector start = new PVector(startFeature.getFloat(0), startFeature.getFloat(1));
  PVector end = new PVector(endFeature.getFloat(0), endFeature.getFloat(1));
  PVector delta = PVector.sub(end, start).div(totalSteps);

  PVector interpData = PVector.add(start, PVector.mult(delta, currStep));

  JSONArray interpCoords = new JSONArray();
  interpCoords.setFloat(0, interpData.x);
  interpCoords.setFloat(1, interpData.y); 

  return interpCoords;
}
