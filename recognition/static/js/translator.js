const video = document.getElementById('video');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]').value;

let category = 'words'; // 기본 카테고리를 'words'로 설정
let lastPredictions = [];
let samePredictionCount = 0;
const minPredictionCount = 5; // 최소 예측 일치 횟수

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing the webcam: ', err);
    });

// Mediapipe 설정
const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

const camera = new Camera(video, {
    onFrame: async () => {
        await hands.send({image: video});
    },
    width: 640,
    height: 480
});
camera.start();

let handsResults = null;

hands.onResults((results) => {
    handsResults = results;
    drawResults();
});

function drawResults() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (handsResults && handsResults.multiHandLandmarks) {
        handsResults.multiHandLandmarks.forEach((landmarks) => {
            drawLandmarks(ctx, landmarks, {color: 'white', lineWidth: 2, radius: 2});
        });

        const landmarksArray = handsResults.multiHandLandmarks.flat().map(landmark => [landmark.x, landmark.y, landmark.z]);
        if (landmarksArray.length > 0) {
            sendLandmarks(landmarksArray);
        }
    } else {
        document.getElementById('result').innerText = 'No hands detected';
    }
}

function drawLandmarks(ctx, landmarks, {color, lineWidth, radius}) {
    for (let i = 0; i < landmarks.length; i++) {
        const x = landmarks[i].x * canvas.width;
        const y = landmarks[i].y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * 3.14);
        ctx.fillStyle = color;
        ctx.fill();
    }
}

function sendLandmarks(landmarks) {
    fetch('/recognition/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ landmarks: landmarks, category: category })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        const topPredictions = data.classes;
        const topProbabilities = data.probabilities;
        const finalPrediction = data.final_prediction;

        if (JSON.stringify(topPredictions) === JSON.stringify(lastPredictions)) {
            samePredictionCount++;
        } else {
            samePredictionCount = 0;
        }

        if (samePredictionCount >= minPredictionCount) {
            document.getElementById('result').innerText = `Top Predictions: ${topPredictions[0]} (${topProbabilities[0].toFixed(2)}), ${topPredictions[1]} (${topProbabilities[1].toFixed(2)})\nFinal Prediction: ${finalPrediction}`;
            samePredictionCount = 0; // Reset count after updating result
        }

        lastPredictions = topPredictions;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Prediction error';
    });
}