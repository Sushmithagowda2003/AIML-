import cv2
import dlib
import numpy as np
import argparse
import os


def load_aadhaar_images(aadhaar_images_dir):
    aadhaar_images = []
    for filename in os.listdir(aadhaar_images_dir):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
            img_path = os.path.join(aadhaar_images_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                aadhaar_images.append(img)
    return aadhaar_images


def load_query_image(query_image_path):
    query_image = cv2.imread(query_image_path)
    return query_image


def initialize_detector_and_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.isfile(predictor_path):
        raise FileNotFoundError(f"Shape predictor file '{predictor_path}' not found.")
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


def detect_faces_dlib(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces


def extract_face_embeddings(image, face, predictor, face_rec_model):
    shape = predictor(image, face)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)


def calculate_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def main(aadhaar_images_dir, query_image_path, similarity_threshold=0.7):
    
    aadhaar_images = load_aadhaar_images(aadhaar_images_dir)
    if not aadhaar_images:
        raise ValueError(f"No valid images found in directory '{aadhaar_images_dir}'")

    query_image = load_query_image(query_image_path)
    if query_image is None:
        raise ValueError(f"Failed to load query image from '{query_image_path}'")

    
    detector, predictor = initialize_detector_and_predictor()

    
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    
    query_faces = detect_faces_dlib(query_image, detector)

   
    query_embeddings = []
    for face in query_faces:
        query_embeddings.append(extract_face_embeddings(query_image, face, predictor, face_rec_model))

    
    matches = 0
    total_aadhaar_images = len(aadhaar_images)

    
    for aadhaar_img in aadhaar_images:
        
        aadhaar_faces = detect_faces_dlib(aadhaar_img, detector)

        
        aadhaar_embeddings = []
        for face in aadhaar_faces:
            aadhaar_embeddings.append(extract_face_embeddings(aadhaar_img, face, predictor, face_rec_model))

       
        for query_emb in query_embeddings:
            for aadhaar_emb in aadhaar_embeddings:
                similarity = calculate_similarity(query_emb, aadhaar_emb)
                
                if similarity > similarity_threshold:
                    matches += 1
                    break  
            if similarity > similarity_threshold:
                break  
    
    accuracy = (matches / total_aadhaar_images) * 100

   
    print(f"Number of Aadhaar card images matching with query image: {matches}")
    print(f"Accuracy: {accuracy}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition for Aadhaar card images.")
    parser.add_argument("aadhaar_images_dir", type=str, help="Path to directory containing Aadhaar card images.")
    parser.add_argument("query_image_path", type=str, help="Path to query image for face recognition.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold for face recognition.")
    args = parser.parse_args()

    main(args.aadhaar_images_dir, args.query_image_path, similarity_threshold=args.threshold)
