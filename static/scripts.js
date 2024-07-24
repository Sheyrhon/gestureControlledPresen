document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            document.getElementById('message').textContent = data.success;
        } else {
            document.getElementById('message').textContent = data.error;
        }
    })
    .catch(error => {
        document.getElementById('message').textContent = 'Error uploading file: ' + error.message;
        console.error('There was an error:', error);
    });
});
