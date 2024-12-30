// Handle form submission for colorization
document.getElementById("colorize-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent default form submission

    const fileInput = document.getElementById("file-input").files[0];
    const loader = document.getElementById("loader");
    const colorizeButton = document.getElementById("colorizeBtn");
    const colorizedImage = document.getElementById("colorized-image");
    const message = document.getElementById("colorize-message");

    if (!fileInput) {
        alert("Please upload an image.");
        return;
    }

    // Prepare the form data
    const formData = new FormData();
    formData.append("image", fileInput);

    // UI updates: show loader and disable the button
    loader.style.display = "flex";
    colorizeButton.disabled = true;
    colorizeButton.textContent = "Processing...";

    try {
        // Send the POST request to the FastAPI endpoint
        const response = await fetch("/colorize", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            // Display the colorized image
            colorizedImage.src = `data:image/png;base64,${data.image_data}`;
            colorizedImage.style.display = "block";
            message.style.display = "block"; // Show success message
        } else {
            // Handle errors returned from the server
            const errorData = await response.json();
            alert(`Error: ${errorData.detail || "An error occurred during processing."}`);
        }
    } catch (error) {
        // Handle network or unexpected errors
        console.error("Error:", error);
        alert(`An unexpected error occurred: ${error.message || "Please try again later."}`);
    } finally {
        // Reset UI: hide loader and re-enable the button
        loader.style.display = "none";
        colorizeButton.disabled = false;
        colorizeButton.textContent = "Upload and Colorize";
    }
});

// Image preview functionality
document.getElementById("file-input").addEventListener("change", function (event) {
    const file = event.target.files[0];
    const previewImage = document.getElementById("preview-image");

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewImage.style.display = "block"; // Show image preview
        };
        reader.readAsDataURL(file);
    } else {
        previewImage.style.display = "none"; // Hide preview if no file selected
    }
});
