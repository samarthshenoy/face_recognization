# import cv2 as cv
# import argparse
# import numpy as np
# import os

# def visualize(image, faces, thickness=2):
#     for idx, face in enumerate(faces[1]):
#         coords = face[:-1].astype(np.int32)
#         cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
#         cv.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
#         cv.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
#         cv.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
#         cv.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
#         cv.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

# def process_image(image_path, faceDetector, recognizer, ref_face_features):
#     image = cv.imread(image_path)
#     if image is None:
#         print(f"Failed to read image: {image_path}")
#         return None, None

#     faceDetector.setInputSize((image.shape[1], image.shape[0]))
#     faces = faceDetector.detect(image)
#     visualize(image, faces)
#     if faces[1] is None:
#         print(f"No faces detected in {image_path}")
#         return None, None

#     face_align = recognizer.alignCrop(image, faces[1][0])
#     face_feature = recognizer.feature(face_align)
    
#     cosine_score = recognizer.match(ref_face_features, face_feature, cv.FaceRecognizerSF_FR_COSINE)
#     l2_score = recognizer.match(ref_face_features, face_feature, cv.FaceRecognizerSF_FR_NORM_L2)
    
#     return cosine_score, l2_score

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-r", "--reference_image", required=True, help="Path to input Aadhaar reference image")
#     ap.add_argument("-d", "--directory", required=True, help="Path to directory with sample images")
#     args = vars(ap.parse_args())
    
#     ref_image = cv.imread(args["reference_image"])
#     if ref_image is None:
#         print("Failed to read reference image")
#         return

#     score_threshold = 0.9
#     nms_threshold = 0.3
#     top_k = 5000
    
#     faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (ref_image.shape[1], ref_image.shape[0]), score_threshold, nms_threshold, top_k)
#     faceInAdhaar = faceDetector.detect(ref_image)
#     if faceInAdhaar[1] is None:
#         print("No faces detected in reference image")
#         return
    
#     visualize(ref_image, faceInAdhaar)
#     cv.imshow("Reference Face", ref_image)
#     cv.waitKey(0)

#     recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")
#     face1_align = recognizer.alignCrop(ref_image, faceInAdhaar[1][0])
#     ref_face_features = recognizer.feature(face1_align)

#     cosine_similarity_threshold = 0.363
#     l2_similarity_threshold = 1.128

#     total_images = 0
#     correct_cosine_matches = 0
#     correct_l2_matches = 0
    
#     for filename in os.listdir(args["directory"]):
#         image_path = os.path.join(args["directory"], filename)
#        cosine_score, l2_score = process_image(image_path, faceDetector, recognizer, ref_face_features)
#         if cosine_score is not None and l2_score is not None:
#             total_images += 1
#             if cosine_score >= cosine_similarity_threshold:
#                 correct_cosine_matches += 1
#             if l2_score <= l2_similarity_threshold:
#                 correct_l2_matches += 1
    
#     if total_images > 0:
#         cosine_accuracy = (correct_cosine_matches / total_images) * 100
#         l2_accuracy = (correct_l2_matches / total_images) * 100
#         print(f"Cosine Similarity Accuracy: {cosine_accuracy:.2f}%")
#         print(f"L2 Similarity Accuracy: {l2_accuracy:.2f}%")
#     else:
#         print("No valid images processed.")

# if __name__ == "__main__":
#     main()
