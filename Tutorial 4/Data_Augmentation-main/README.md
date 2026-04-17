# Data Augmentation Using Python

## Objectives
- Understand the need for data augmentation in deep learning.
- Understand different image augmentation operations.
- Implement image augmentation using Python.

## Introduction
Data augmentation is an important technique in deep learning. Since collecting large datasets is difficult, augmentation helps increase dataset size by applying transformations to existing images. It improves model performance and reduces overfitting by introducing variability.

## Augmentation Techniques Used

The following augmentation operations were applied:

- Random Rotation (±40 degrees)
- Random Affine Transformation (Shearing effect)
- Random Resized Crop (Zooming effect)
- Random Horizontal Flip
- Color Jitter (Brightness adjustment)

## Implementation Steps

### Step 1: Import Required Libraries
Libraries used:
- os
- PIL (Python Imaging Library)
- torchvision.transforms
- matplotlib.pyplot

### Step 2: Define Image Path and Output Folder
- A single JPEG image was selected from the computer.
- A folder was created to save augmented images.

### Step 3: Apply Data Augmentation
- `transforms.Compose()` was used to combine multiple augmentation techniques.
- The following transformations were applied:
  - Rotation  
  - Shearing  
  - Zooming  
  - Horizontal flipping  
  - Brightness adjustment  

### Step 4: Generate and Save Augmented Images
- 40 augmented images were generated from the original image.
- Each augmented image was saved in the specified directory.
- Images were also displayed using matplotlib.

## Output Location
The augmented images were savedin augmented_image folder.

## Conclusion
Data augmentation helps improve deep learning model generalization by increasing dataset diversity and reducing overfitting.
