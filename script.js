import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
import kNear from "./kNear.js";


// nn
ml5.setBackend("webgl");

const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

const demosSection = document.getElementById("demos");
const k = 3
const machine = new kNear(k);
let poseLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "20vw";
const videoWidth = "20vw";
let poseLandmarks; // Je pose data komt hieruit
let startGameButton = document.getElementById("start-game")
// let nextQuestionButton = document.getElementById("nextQuestion")
// nextQuestionButton.addEventListener("click", loadQuestion)
let prediction = null;
let questionCount = 0
let answerButtons = document.querySelectorAll('.answer-btn');
let predictionImage = document.getElementById("result")




startGameButton.addEventListener("click", startGame)
window.addEventListener("load", loadPoseData)
let countdownEl = document.getElementById("countdown-number")
let countdownCircle = document.getElementById("circle")
let correctText = document.getElementById("correct-text")

function loadPoseData() {
    fetch('trainingData.json')
        .then(response => response.json())
        .then(data => {
            let storedData = data;
            console.log(storedData); // check of de data klopt

            for (let label in storedData) {
                storedData[label].forEach(pose => {
                    machine.learn(pose, label);
                });
            }
        })
        .catch(error => {
            console.error('Fout bij het laden van JSON:', error);
        });

    const modelDetails = {
        model: 'model/model.json',
        metadata: 'model/model_meta.json',
        weights: 'model/model.weights.bin'
    }
    nn.load(modelDetails, () => console.log("het model is geladen!"))
}

// Load the PoseLandmarker
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 2
    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();

// Handle click detection on images
const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!poseLandmarker) {
        console.log("Wait for poseLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await poseLandmarker.setOptions({ runningMode: "IMAGE" });
    }

    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (let i = allCanvas.length - 1; i >= 0; i--) {
        allCanvas[i].parentNode.removeChild(allCanvas[i]);
    }

    poseLandmarker.detect(event.target, (result) => {
        const canvas = document.createElement("canvas");
        canvas.setAttribute("class", "canvas");
        canvas.setAttribute("width", event.target.naturalWidth + "px");
        canvas.setAttribute("height", event.target.naturalHeight + "px");
        canvas.style =
            "left: 0px;" +
            "top: 0px;" +
            "width: " +
            event.target.width +
            "px;" +
            "height: " +
            event.target.height +
            "px;";

        event.target.parentNode.appendChild(canvas);
        const canvasCtx = canvas.getContext("2d");
        // const drawingUtils = new DrawingUtils(canvasCtx);
        // for (const landmark of result.landmarks) {
        //     drawingUtils.drawLandmarks(landmark, {
        //         radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        //     });
        //     drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        // }
    });
}

// Webcam detection
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    startGameButton.style = "display: flex"

    webcamRunning = !webcamRunning;
    enableWebcamButton.style = "display: none"
    correctText.innerHTML = "Hieronder zie je de houding die JIJ doet"

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            if(result.landmarks && result.landmarks.length>0) {
                poseLandmarks = result.landmarks
                let flattenedResults = Object.values(result.landmarks[0]).flatMap(landmark => [
                    landmark.x, landmark.y, landmark.z
                ]);
                predictResults(flattenedResults)

            }
            // for (const landmark of result.landmarks) {
            //     drawingUtils.drawLandmarks(landmark, {
            //         radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
            //     });
            //     drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            // }
            canvasCtx.restore();
        });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}
import { sommen } from './mathProblems.js';

// window.addEventListener('DOMContentLoaded', () => {
//
// });
function startGame(){
    startGameButton.style = "display: none"
    answerButtons.forEach(answerButton => {
        answerButton.style.display = "flex";
    });

    loadQuestion()
}
function loadQuestion(){
    predictionImage.style = "display:flex"
    // nextQuestionButton.style = "display: none"
    const questionEl = document.getElementById('question');

    // Pak de eerste vraag
    const firstProblem = sommen[questionCount];

    // Toon de som
    questionEl.textContent = firstProblem.question;

    // Zet de opties en de images
    firstProblem.options.forEach((option, index) => {
        const button = answerButtons[index];
        const img = button.querySelector('img');

        button.innerHTML = `${option}
      <img src="./images/${firstProblem.poses[index]}.png" alt="${firstProblem.poses[index]} Pose" class="pose-image" />
      
    `;
    });

    let count = 10;
    countdownEl.textContent = count;
    const countdownInterval = setInterval(() => {
        count--;
        if (count > 0) {
            countdownEl.textContent = count;
        } else {
            clearInterval(countdownInterval);
            countdownEl.textContent = "";

            questionEl.style.visibility = "visible";
            answerButtons.forEach(button => button.style.visibility = "visible");

            // Check antwoord na 1 extra seconde om zeker te zijn van prediction
            setTimeout(() => {
                checkAnswer(firstProblem);
            }, 1500);
        }

    }, 1000);
    questionCount++
}

function checkAnswer(currentProblem) {
    if (!prediction) {
        console.log("❌ Geen prediction beschikbaar.");
        return;
    }

    const correctPose = currentProblem.poses[currentProblem.options.indexOf(currentProblem.correct)];
    console.log("Juiste pose moet zijn:", correctPose);
    predictionImage.style = "display:none"
    if (prediction === correctPose) {
        console.log("✅ Correcte pose!");
        correctText.innerHTML = `✅Correct het juiste antwoord is: ${currentProblem.correct}`

    } else {
        console.log("❌ Verkeerde pose:", prediction);
        correctText.innerHTML = `❌Fout het juiste antwoord is: ${currentProblem.correct}`

    }
    setTimeout(() => {
        correctText.innerHTML = ""
        loadQuestion()
    }, 1000);
}

async function predictResults(data) {
    let predictionImage = document.getElementById("result")

    const results = await nn.classify(data)

    // prediction = machine.classify(
    //     data
    // )
    prediction = results[0].label
    // console.log(`I think this is a ${prediction}`)
    if (prediction) {
        predictionImage.src = "./images/" + results[0].label + ".png"
        console.log(prediction)
    }
    // if (results) {
    //     predictionImage.src = "./images/" + results[0].label + ".png"
    //     console.log(results[0].label)
    // }
}

