<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Virtual Whiteboard</title>
<style>
body {
  margin: 0;
  overflow: hidden;
  background: black;
  font-family: Arial;
}

#video {
  position: absolute;
  top: 0;
  left: 0;
  width: 1280px;
  height: 720px;
  object-fit: cover;
  transform: scaleX(-1);
}

#canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 1280px;
  height: 720px;
  z-index: 10;
}

#toolbar {
  position: absolute;
  top: 0;
  left: 0;
  height: 70px;
  width: 1280px;
  background: #111;
  z-index: 20;
  display: flex;
  align-items: center;
  padding: 5px;
  gap: 10px;
}

.btn {
  padding: 6px 10px;
  background: #333;
  color: white;
  border: none;
  cursor: pointer;
}

.color {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 2px solid white;
  cursor: pointer;
}
</style>
</head>

<body>

<video id="video" autoplay></video>
<canvas id="canvas"></canvas>

<div id="toolbar">
  <button class="btn" onclick="undo()">Undo</button>
  <button class="btn" onclick="clearCanvas()">Clear</button>

  <div class="color" style="background:white" onclick="setColor('white')"></div>
  <div class="color" style="background:black" onclick="setColor('black')"></div>
  <div class="color" style="background:red" onclick="setColor('red')"></div>
  <div class="color" style="background:blue" onclick="setColor('blue')"></div>

  <button class="btn" onclick="setSize(3)">3</button>
  <button class="btn" onclick="setSize(6)">6</button>
  <button class="btn" onclick="setSize(10)">10</button>
  <button class="btn" onclick="setSize(20)">20</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = 1280;
canvas.height = 720;

let color = "red";
let size = 6;
let drawing = false;
let last = null;

let strokes = [];
let currentStroke = [];

function setColor(c){ color = c; }
function setSize(s){ size = s; }

function undo(){
  strokes.pop();
  redraw();
}

function clearCanvas(){
  strokes = [];
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

function drawLine(a,b,c,s){
  ctx.strokeStyle = c;
  ctx.lineWidth = s;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x,a.y);
  ctx.lineTo(b.x,b.y);
  ctx.stroke();
}

function redraw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let s of strokes){
    for(let i=1;i<s.length;i++){
      drawLine(s[i-1], s[i], s.color, s.size);
    }
  }
}

const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(onResults);

function onResults(results){
  if(!results.multiHandLandmarks) return;

  const lm = results.multiHandLandmarks[0];

  const index = lm[8];
  const thumb = lm[4];

  const x = index.x * canvas.width;
  const y = index.y * canvas.height;

  const pinch = Math.hypot(
    index.x - thumb.x,
    index.y - thumb.y
  ) < 0.05;

  if(pinch){
    drawing = true;

    if(!currentStroke.length){
      currentStroke = [];
      strokes.push(currentStroke);
      currentStroke.color = color;
      currentStroke.size = size;
    }

    currentStroke.push({x,y});

    if(last){
      drawLine(last, {x,y}, color, size);
    }

    last = {x,y};
  } else {
    drawing = false;
    last = null;
    currentStroke = [];
  }
}

const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({image: video});
  },
  width: 1280,
  height: 720
});

camera.start();
</script>

</body>
</html>
