import os
import cv2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to create the DenseNet169 model
def create_model(input_shape=(128, 128, 3), num_classes=2):
    base_model = DenseNet169(weights=None, include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to capture video and save frames
def capture_video_and_save_frames(person_name, output_dir="frames", frame_count=200, fps=20):
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Recording video for {person_name}. Press 'q' to stop early.")
    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        cv2.imshow(f"Recording for {person_name}", frame)
        frame_path = os.path.join(person_dir, f"{person_name}_{frame_idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("Stopped recording early.")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video recording for {person_name} complete. Frames saved to {person_dir}.")

# Function to preprocess images
def preprocess_frames(input_dir, output_dir, target_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    for person_name in os.listdir(input_dir):
        person_input_dir = os.path.join(input_dir, person_name)
        person_output_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_output_dir, exist_ok=True)
        for file_name in os.listdir(person_input_dir):
            file_path = os.path.join(person_input_dir, file_name)
            if file_name.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(file_path)
                img_resized = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(person_output_dir, file_name), img_resized)
    print(f"Preprocessed frames saved to {output_dir}.")

# Function to train the model
def train_model(dataset_dir, target_size=(128, 128), batch_size=32, epochs=10):
    current_dir = os.getcwd()
    model_save_path = os.path.join(current_dir, "trained_model.h5")
    datagen = ImageDataGenerator(validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'
    )
    num_classes = len(train_gen.class_indices)
    model = create_model(num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model and store the training history
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

   # Save the history as a JSON file
    with open('history.json', 'w') as f:
     json.dump(history.history, f)  # Properly indented
    print("Training history saved to history.json")  # This line is outside the with block


# Return the trained model and history
    return model, history


# Main script
if __name__ == "__main__":
    while True:
        add_new_person = input("Do you want to add a new person? (yes/no): ").strip().lower()

        if add_new_person == "yes":
            # Delete the existing model
            model_path = "trained_model.h5"
            if os.path.exists(model_path):
                os.remove(model_path)
                print("Existing model deleted.")

            # Add a new person
            person_name = input("Enter the name of the person: ").strip()
            capture_video_and_save_frames(person_name)
            preprocess_frames("frames", "preprocessed")
            print(f"New person {person_name} enrolled successfully!")

            # Train a new model
            print("Training a new model...")
            train_model("preprocessed", epochs=10)

            # Ask to add more people
            add_more = input("Do you want to add another person? (yes/no): ").strip().lower()
            if add_more != "yes":
                break

        elif add_new_person == "no":
            # Ask if user wants to retrain the model
            retrain = input("Do you want to retrain the model? (yes/no): ").strip().lower()
            if retrain == "yes":
                print("Retraining the model...")
                train_model("preprocessed", epochs=10)
            else:
                print("No action performed. Exiting.")
            break

        else:
            print("Invalid input. Please type 'yes' or 'no'.")
