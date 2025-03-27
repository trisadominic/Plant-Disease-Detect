document.getElementById("imageUpload").addEventListener("change", function (event) {
    let file = event.target.files[0];
    if (file) {
        let reader = new FileReader();
        reader.onload = function (e) {
            let previewImage = document.getElementById("previewImage");
            previewImage.src = e.target.result;
            previewImage.style.display = "block"; // Show preview
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById("analyzeBtn").addEventListener("click", function () {
    let formData = new FormData();
    let imageFile = document.getElementById("imageUpload").files[0];

    if (!imageFile) {
        alert("❌ Please upload an image!");
        return;
    }

    formData.append("file", imageFile);

    fetch("/predict", {  // Using relative URL
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultElement = document.getElementById("result");
        if (data.error) {
            resultElement.innerText = "❌ Error: " + data.error;
        } else {
            resultElement.innerText = `✅ Disease: ${data.disease} | Confidence: ${data.confidence}`;
        }
    })
    .catch(error => {
        document.getElementById("result").innerText = "❌ Fetch Error: " + error;
    });
});
