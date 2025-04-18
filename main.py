import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import speech_recognition as sr
# from transformers import pipeline # Keep commented unless needed and installed

# --- Initialization ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

engine = pyttsx3.init()
engine.setProperty('rate', 150)

r = sr.Recognizer()
r.energy_threshold = 4000 # Adjust based on your microphone sensitivity
r.dynamic_energy_threshold = True # Adjust energy threshold dynamically

# --- Constants for Exercises ---
# Bicep Curl
BICEP_CURL_ANGLE_UP = 160
BICEP_CURL_ANGLE_DOWN = 30
BICEP_CURL_FEEDBACK_THRESHOLD = 120 # Angle above which elbow might be flaring

# Squat
SQUAT_ANGLE_UP = 165
SQUAT_ANGLE_DOWN = 90 # Target depth angle for knees
SQUAT_FEEDBACK_DEPTH = 100 # Provide feedback if not going below this

# Triceps Extension (Overhead assumed)
TRICEP_ANGLE_UP = 160 # Extended arm
TRICEP_ANGLE_DOWN = 70  # Bent arm
TRICEP_FEEDBACK_EXTENSION = 150 # Provide feedback if not extending past this

# --- VERY ROUGH Calorie Estimates Per Rep (FOR DEMONSTRATION ONLY) ---
# These are illustrative values and not scientifically accurate.
calories_per_rep_estimate = {
    "bicep_curl": 0.5,       # Low estimate for isolation exercise
    "squat": 1.5,           # Higher estimate for compound exercise
    "tricep_extension": 0.4  # Low estimate for isolation exercise
}

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., shoulder, elbow, wrist)."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point (joint)
    c = np.array(c) # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def listen_for_command():
    """Listens for a voice command using the microphone."""
    with sr.Microphone() as source:
        print("Listening for command (e.g., 'start bicep workout', 'start squat workout', 'start tricep workout', 'stop workout')...")
        # r.adjust_for_ambient_noise(source, duration=0.5) # Adjust for ambient noise
        try:
            audio = r.listen(source, phrase_time_limit=4, timeout=3) # Listen for 4 seconds max, wait 3 seconds
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            # print("No command heard.")
            return ""
        except sr.UnknownValueError:
            # print("Could not understand audio") # Reduced noise
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred during listening: {e}")
            return ""

def calculate_average_rep_time(workout_data):
    """Calculates the average time per rep."""
    if not workout_data or 'reps' not in workout_data or len(workout_data['reps']) <= 1:
        return 0
    try:
        # Ensure timestamps are present
        if 'timestamp' not in workout_data['reps'][0] or 'timestamp' not in workout_data['reps'][-1]:
             return 0
        total_time = workout_data['reps'][-1]['timestamp'] - workout_data['reps'][0]['timestamp']
        num_reps = len(workout_data['reps'])
        # Calculate time over the number of intervals between reps
        return total_time / (num_reps - 1) if num_reps > 1 else 0
    except KeyError:
        print("Error calculating average rep time: Timestamp key missing.")
        return 0
    except ZeroDivisionError:
        return 0


def get_form_deviation_details(workout_data, exercise_type):
    """Checks for form deviations based on the exercise type."""
    deviations = {}
    if not workout_data or 'reps' not in workout_data:
        return deviations

    reps_data = workout_data['reps']
    if exercise_type == "bicep_curl":
        elbow_out_count = sum(1 for rep in reps_data if rep.get('elbow_out', False))
        deviations["elbow_out_count"] = elbow_out_count
    elif exercise_type == "squat":
        shallow_reps = sum(1 for rep in reps_data if rep.get('shallow_depth', False))
        deviations["shallow_reps"] = shallow_reps
    elif exercise_type == "tricep_extension":
        incomplete_extension = sum(1 for rep in reps_data if rep.get('incomplete_extension', False))
        deviations["incomplete_extension"] = incomplete_extension
    # Add more exercise-specific checks here
    return deviations


def generate_report_template(workout_data, analysis_results, exercise_type):
    """Generates a text report based on workout data and analysis."""
    if not workout_data or 'reps' not in workout_data:
        return f"No workout data recorded for {exercise_type.replace('_', ' ')}."

    rep_count = analysis_results.get("total_reps", 0) # Get total reps from analysis results
    if rep_count == 0:
        return f"You completed 0 reps of {exercise_type.replace('_', ' ')}."

    time_per_rep = analysis_results.get("average_time_per_rep", 0)
    exercise_name = exercise_type.replace('_', ' ').title()
    est_calories = analysis_results.get("estimated_calories_burned", 0) # Get estimated calories

    form_message = f"Overall form analysis for {exercise_name}: "
    deviation_found = False

    if exercise_type == "bicep_curl":
        elbow_deviation_count = analysis_results.get('elbow_out_count', 0)
        if elbow_deviation_count > 0:
            form_message += f"You kept your elbow out {elbow_deviation_count} times. Focus on keeping elbows tucked in. "
            deviation_found = True
    elif exercise_type == "squat":
        shallow_reps_count = analysis_results.get('shallow_reps', 0)
        if shallow_reps_count > 0:
            form_message += f"You didn't reach full depth on {shallow_reps_count} reps. Try to squat lower. "
            deviation_found = True
    elif exercise_type == "tricep_extension":
        incomplete_extension_count = analysis_results.get('incomplete_extension', 0)
        if incomplete_extension_count > 0:
            form_message += f"You didn't fully extend your arm on {incomplete_extension_count} reps. Focus on full lockout. "
            deviation_found = True

    if not deviation_found and rep_count > 0: # Added check for rep_count > 0
        form_message += "Good job maintaining form!"
    elif rep_count == 0:
        form_message = "" # No form message if no reps completed

    # Updated report string to include estimated calories
    return (f"Workout Summary for {exercise_name}: You completed {rep_count} reps, "
            f"burning an estimated {est_calories:.1f} calories. "
            f"Your average time per rep was {time_per_rep:.2f} seconds. {form_message}")


# --- Option C: LLM Report Generation (Optional) ---
# def generate_report_with_model(analysis_results, exercise_type):
#     # ... (LLM code remains the same, just ensure analysis_results contains calorie data if needed)

# --- Main Execution ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State variables
counter = 0
stage = None # e.g., "up", "down"
current_exercise = None # e.g., "bicep_curl", "squat", "tricep_extension"
workout_active = False
estimated_calories_burned = 0.0 # Initialize calorie counter
last_feedback_time = 0
last_feedback_type = "" # To avoid repeating the same feedback
workout_data = {} # Reset at the start of each workout

llm_report_generation = False # Set to True to use LLM report (requires transformers)
frame_counter = 0 # For skipping frames to potentially improve performance
draw_counter = 0 # For skipping drawing to potentially improve performance

last_command_check_time = time.time()
COMMAND_CHECK_INTERVAL = 1.5 # Check for voice commands every 1.5 seconds

print("Initializing...")
# Give pyttsx3 time to initialize if needed
time.sleep(0.5)
engine.say("System Ready. Say 'start' followed by the exercise name to begin.")
engine.runAndWait()


with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Reduce processing load by skipping frames
        frame_counter += 1
        if frame_counter % 2 != 0: # Process every 2nd frame
            continue

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check for command periodically
        current_time = time.time()
        command = ""
        if current_time - last_command_check_time > COMMAND_CHECK_INTERVAL:
             command = listen_for_command()
             last_command_check_time = current_time # Reset timer even if no command heard

        # Command Handling
        if not workout_active:
            exercise_detected = None
            if "start bicep workout" in command or "start bicep curl" in command :
                 exercise_detected = "bicep_curl"
            elif "start squat workout" in command or "start squats" in command:
                 exercise_detected = "squat"
            elif "start tricep workout" in command or "start triceps" in command:
                 exercise_detected = "tricep_extension"

            if exercise_detected:
                current_exercise = exercise_detected
                workout_active = True
                workout_data = { # Reset workout data for the new session
                    'start_time': time.time(),
                    'end_time': None,
                    'reps': [],
                    'feedback': [],
                    'exercise_type': current_exercise
                }
                counter = 0
                stage = None
                estimated_calories_burned = 0.0 # <<< RESET CALORIES ON START
                last_feedback_time = 0
                last_feedback_type = ""
                print(f"Workout started: {current_exercise.replace('_', ' ').title()}")
                engine.say(f"{current_exercise.replace('_', ' ').title()} Workout Started")
                engine.runAndWait()

        elif "stop workout" in command and workout_active:
            workout_data['end_time'] = time.time()
            print(f"Workout stopped: {current_exercise.replace('_', ' ').title()}")
            engine.say("Workout stopped")
            engine.runAndWait()

            # --- Generate Report ---
            analysis_results = {}
            avg_time = calculate_average_rep_time(workout_data)
            analysis_results["average_time_per_rep"] = avg_time
            deviation_details = get_form_deviation_details(workout_data, current_exercise)
            analysis_results.update(deviation_details)
            analysis_results["total_reps"] = len(workout_data.get('reps', []))
            # --- ADD ESTIMATED CALORIES TO RESULTS ---
            analysis_results["estimated_calories_burned"] = estimated_calories_burned
            # --- END ADDITION ---

            report = ""
            if llm_report_generation:
                 # report = generate_report_with_model(analysis_results, current_exercise)
                 print("LLM Reporting currently disabled. Using template.")
                 report = generate_report_template(workout_data, analysis_results, current_exercise)
            else:
                report = generate_report_template(workout_data, analysis_results, current_exercise)

            print("\n--- Workout Report ---")
            print(report)
            print("----------------------\n")
            # Ensure report is not empty before saying it
            if report:
                engine.say(report)
                engine.runAndWait()

            # Reset state after stopping
            workout_active = False
            current_exercise = None
            workout_data = {}
            counter = 0
            stage = None
            estimated_calories_burned = 0.0 # Reset calories display


        # Workout Logic - Only if workout is active and landmarks are detected
        if workout_active and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            current_rep_info = {"timestamp": time.time()} # Base info for the rep
            feedback_message = ""
            feedback_type_this_frame = "" # What kind of feedback to potentially give

            try:
                # --- BICEP CURL LOGIC ---
                if current_exercise == "bicep_curl":
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Curl counter logic
                    if angle > BICEP_CURL_ANGLE_UP:
                        stage = "down"
                    if angle < BICEP_CURL_ANGLE_DOWN and stage == 'down':
                        stage = "up"
                        counter += 1
                        # --- ADD CALORIE CALCULATION ---
                        estimated_calories_burned += calories_per_rep_estimate.get(current_exercise, 0)
                        # --- END ADDITION ---
                        current_rep_info['type'] = 'bicep_curl'
                        current_rep_info['end_angle'] = angle # Store final angle of the rep
                        workout_data['reps'].append(current_rep_info)
                        print(f"Rep: {counter} | Est. Cal: {estimated_calories_burned:.1f}") # Log calories too
                        # engine.say(str(counter)) # Optional

                    # Form feedback: Elbow flaring out
                    if stage == 'up' and angle > BICEP_CURL_FEEDBACK_THRESHOLD:
                         feedback_message = "Keep elbow tucked in"
                         feedback_type_this_frame = "elbow_out"
                         if workout_data['reps']:
                             workout_data['reps'][-1]['elbow_out'] = True


                # --- SQUAT LOGIC ---
                elif current_exercise == "squat":
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate knee angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Squat counter logic
                    if angle > SQUAT_ANGLE_UP:
                        if stage == 'down':
                            min_angle_this_rep = current_rep_info.get('min_angle', angle) # Get min angle recorded during down phase
                            # Check depth based on the lowest point reached
                            if min_angle_this_rep > SQUAT_ANGLE_DOWN:
                                current_rep_info['shallow_depth'] = True # Mark *this* rep as shallow

                            counter += 1
                            # --- ADD CALORIE CALCULATION ---
                            estimated_calories_burned += calories_per_rep_estimate.get(current_exercise, 0)
                            # --- END ADDITION ---
                            current_rep_info['type'] = 'squat'
                            current_rep_info['end_angle'] = angle
                            workout_data['reps'].append(current_rep_info)
                            print(f"Rep: {counter} | Est. Cal: {estimated_calories_burned:.1f}") # Log calories too
                            # engine.say(str(counter)) # Optional

                        stage = "up" # Now fully up

                    if angle < SQUAT_ANGLE_DOWN and stage != 'down': # Use stage != 'down' to trigger once
                        stage = "down"
                        # Reset min angle tracking for the new rep starting down
                        current_rep_info['min_angle'] = angle

                    # Update min angle while in 'down' stage
                    if stage == 'down':
                        current_rep_info['min_angle'] = min(angle, current_rep_info.get('min_angle', angle))

                    # Form feedback: Depth check (provide feedback if currently shallow)
                    if stage == 'down' and angle > SQUAT_FEEDBACK_DEPTH:
                        feedback_message = "Go deeper"
                        feedback_type_this_frame = "squat_depth"


                # --- TRICEPS EXTENSION LOGIC ---
                elif current_exercise == "tricep_extension":
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate elbow angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Triceps counter logic
                    if angle < TRICEP_ANGLE_DOWN and stage != 'down': # Use stage != 'down' to trigger once
                        stage = "down" # Arm is bent
                        # Reset max angle tracking
                        current_rep_info['max_angle'] = angle

                    if angle > TRICEP_ANGLE_UP and stage == 'down':
                        stage = "up" # Arm is extended
                        max_angle_this_rep = current_rep_info.get('max_angle', angle)
                        # Check extension based on max angle reached
                        if max_angle_this_rep < TRICEP_ANGLE_UP:
                             current_rep_info['incomplete_extension'] = True
                        else:
                             current_rep_info['incomplete_extension'] = False

                        counter += 1
                        # --- ADD CALORIE CALCULATION ---
                        estimated_calories_burned += calories_per_rep_estimate.get(current_exercise, 0)
                        # --- END ADDITION ---
                        current_rep_info['type'] = 'tricep_extension'
                        current_rep_info['end_angle'] = angle # Store the final extended angle
                        workout_data['reps'].append(current_rep_info)
                        print(f"Rep: {counter} | Est. Cal: {estimated_calories_burned:.1f}") # Log calories too
                        # engine.say(str(counter)) # Optional

                    # Update max angle while moving towards 'up' stage
                    if stage == 'up' or (stage == 'down'): # Update max angle in both down and up stages before counting rep
                        current_rep_info['max_angle'] = max(angle, current_rep_info.get('max_angle', angle))

                    # Form feedback: Full extension check (provide feedback if currently not extended enough)
                    if stage == 'up' and angle < TRICEP_FEEDBACK_EXTENSION:
                         feedback_message = "Extend arm fully"
                         feedback_type_this_frame = "tricep_extension"
                         # Mark the potential issue for the rep being formed/just completed
                         if workout_data['reps']:
                            workout_data['reps'][-1]['incomplete_extension'] = True


                # --- Provide Feedback ---
                time_now = time.time()
                if feedback_message and (time_now - last_feedback_time > 3 or feedback_type_this_frame != last_feedback_type):
                    print(f"Feedback: {feedback_message}")
                    engine.say(feedback_message)
                    engine.runAndWait()
                    last_feedback_time = time_now
                    last_feedback_type = feedback_type_this_frame
                    # Log feedback given (Optional: could refine this)
                    if 'feedback' in workout_data:
                        workout_data['feedback'].append({
                            "timestamp": time_now,
                            "type": feedback_type_this_frame,
                            "message": feedback_message
                        })

            except IndexError:
                 # Handle cases where landmarks might not be detected temporarily
                 print("Warning: Could not access required landmarks. Ensure body is fully visible.")
            except Exception as e:
                print(f"Error processing landmarks for {current_exercise}: {e}")


            # --- Visualization ---
            # Status Box (Increased Height)
            box_width = 250 # Wider box can be helpful
            cv2.rectangle(image, (0, 0), (box_width, 100), (245, 117, 16), -1) # Taller box

            # Rep Counter
            cv2.putText(image, 'REPS', (15, 20), # Adjusted Y
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60), # Adjusted Y
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage Display
            cv2.putText(image, 'STAGE', (int(box_width * 0.5), 20), # Adjusted X and Y
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else 'WAIT', # Show WAIT if no stage yet
                        (int(box_width * 0.45), 60), # Adjusted X and Y
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # --- CALORIE DISPLAY ---
            cv2.putText(image, 'EST. CAL', (15, 85), # Position below Reps Label
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, f"{estimated_calories_burned:.1f}",
                        (int(box_width * 0.45), 85), # Align with Stage value approx
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            # --- END CALORIE DISPLAY ---


            # Display Current Exercise Name (Below the box)
            if current_exercise:
                 exercise_display_name = current_exercise.replace('_', ' ').title()
                 cv2.putText(image, exercise_display_name, (10, 125), # Position below the taller box
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        # Render detections - Reduce drawing frequency
        draw_counter += 1
        if draw_counter % 2 == 0: # Draw every 4th original frame
             # Check if pose landmarks exist before drawing
             if results.pose_landmarks:
                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Joints
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Connections
                                         )

        # Display frame
        cv2.imshow('Mediapipe Feed', image)

        # Exit condition
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Application Closed.")
