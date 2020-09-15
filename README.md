# Leaf Disease Detector
 
## SVM Model:

Classifies leaves in 2 classes (healthy and rotten).

Followed Steps:

* Image is transformed to HSV (improve the
RGB model).
* Image segmentation to identify rotten parts of the leaf.
* Feature extraction is performed for every sample that is
available for training.
* Training of the SVM classifier performs, generating two
classes (healthy leaves 1 class, rotten leaves 2 class).

## Datasets

* Datasets consists of apple laf images.
* Training Images: 2000 images (1000 healthy leaf images)
and (1000 rotten leaf images).
* Testing Images: 40 images for testing (20 healthy leaf
images) and (20 rotten leaf images).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.