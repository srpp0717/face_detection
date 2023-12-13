import cv2 as cv

# Load the image
img = cv.imread('005.jpeg')

# Load the face detection model
face_model = cv.CascadeClassifier('face-detect-model.xml')

# Convert the image to grayscale
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the image with adjusted parameters for better accuracy
faces = face_model.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Resize the image to fit the screen
height, width, _ = img.shape
resize_factor = min(1.0, 800.0 / width)
resized_img = cv.resize(img, (int(width * resize_factor), int(height * resize_factor)))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    # Adjust the coordinates and size based on the resize factor
    x, y, w, h = int(x * resize_factor), int(y * resize_factor), int(w * resize_factor), int(h * resize_factor)

    cv.rectangle(resized_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Count the number of faces
num_faces = len(faces)

# Display the number of faces on the image with shadow
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(resized_img, f'Number of Faces: {num_faces}', (10, 30), font, 1, (0, 0, 0), 2, cv.LINE_AA)
cv.putText(resized_img, f'Number of Faces: {num_faces}', (11, 31), font, 1, (255, 255, 255), 2, cv.LINE_AA)

# Display the resized image with the face count and shadow
cv.imshow('Resized Image', resized_img)
cv.waitKey(0)
cv.destroyAllWindows()
