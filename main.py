import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import speech_recognition as sr
from transformers import pipeline  # For the LLM Option

# --- Initialization ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

engine = pyttsx3.init()

engine.setProperty('rate', 150)

r = sr.Recognizer()

# --- Helper Functions ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def listen_for_command():
    with sr.Microphone() as source:
        print("Say something...")
        audio = r.listen(source, phrase_time_limit=3)
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def calculate_average_rep_time(workout_data):
    if len(workout_data['reps']) <= 1:
      return 0
    total_time = workout_data['reps'][-1]['timestamp'] - workout_data['reps'][0]['timestamp']
    return total_time / len(workout_data['reps'])

def check_elbow_form_consistency(workout_data):
    elbow_out_count = 0
    for rep in workout_data['reps']:
        if rep['elbow_out']:
            elbow_out_count += 1
    return elbow_out_count

def get_form_deviation_details(workout_data):
    elbow_out_count = check_elbow_form_consistency(workout_data)
    return {"elbow_out_count": elbow_out_count}

def generate_report_template(workout_data, analysis_results):
    rep_count = len(workout_data['reps'])
    time_per_rep = analysis_results["average_time_per_rep"]
    elbow_deviation_count = analysis_results['elbow_out_count']

    if elbow_deviation_count > 0:
        elbow_form_message = f"You kept your elbow out {elbow_deviation_count} times. Focus on pulling it in."
    else:
        elbow_form_message = "Your elbow form was good."

    return f"You completed {rep_count} reps. Your average time per rep was {time_per_rep:.2f} seconds. {elbow_form_message}"

# --- Option C: LLM Report Generation ---
def generate_report_with_model(analysis_results):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    report_prompt = f"""
            Generate a report based on these results: {analysis_results}.
            Make sure to give feedback, and give a short concise summary of the results.
            """
    summary = summarizer(report_prompt, max_length=300, min_length=30)
    return summary[0]['summary_text']

# --- Main Execution ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
stage = None
workout_active = False
last_feedback_time = 0
last_feedback = ""
workout_data = {
    'start_time': None,
    'end_time': None,
    'reps': [],
    'feedback': []
}
llm_report_generation = False #change this if you would prefer template based or LLM based generation
frame_counter = 0
draw_counter = 0
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        frame_counter +=1
        if frame_counter % 2 !=0:
             continue
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        num_landmarks = 0
        if results.pose_landmarks:
           num_landmarks = len(results.pose_landmarks.landmark)
        # Speech Command Handling
        command = listen_for_command()
        if "start workout" in command and not workout_active:
            workout_active = True
            print("Workout started automatically")
            workout_data['start_time'] = time.time()
            engine.say("Workout Started")
            engine.runAndWait()
            counter = 0
            stage = None
        elif "stop workout" in command and workout_active:
            workout_data['end_time'] = time.time()
            workout_active = False
            print("Workout stopped")
            engine.say("Workout stopped")
            engine.runAndWait()
            analysis_results = {}
            analysis_results["average_time_per_rep"] = calculate_average_rep_time(workout_data)
            deviation_details = get_form_deviation_details(workout_data)
            analysis_results.update(deviation_details)
            if llm_report_generation:
                report = generate_report_with_model(analysis_results)
            else:
              report = generate_report_template(workout_data, analysis_results)
            print("Report Generated:")
            print(report)
            engine.say(report)
            engine.runAndWait()
            workout_data = {
              'start_time': None,
              'end_time': None,
              'reps': [],
              'feedback': []
             } # Reset the workout data
        elif num_landmarks > 20 and not workout_active:
           workout_active = True
           print("Workout started automatically")
           workout_data['start_time'] = time.time()
           engine.say("Workout Started")
           engine.runAndWait()
           counter = 0
           stage = None
        elif num_landmarks <= 20 and workout_active:
           workout_data['end_time'] = time.time()
           workout_active = False
           print("Workout stopped")
           engine.say("Workout stopped")
           engine.runAndWait()
           analysis_results = {}
           analysis_results["average_time_per_rep"] = calculate_average_rep_time(workout_data)
           deviation_details = get_form_deviation_details(workout_data)
           analysis_results.update(deviation_details)
           if llm_report_generation:
                report = generate_report_with_model(analysis_results)
           else:
                report = generate_report_template(workout_data, analysis_results)

           print("Report Generated:")
           print(report)
           engine.say(report)
           engine.runAndWait()
           workout_data = {
              'start_time': None,
              'end_time': None,
              'reps': [],
              'feedback': []
             } # Reset the workout data
        if workout_active and results.pose_landmarks: #Check if landmarks are valid before processing
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    current_time = time.time()
                    workout_data['reps'].append({"timestamp":current_time, "elbow_angle_range":None, "elbow_out": False})
                    print(f"Rep: {counter}")
                    engine.say(str(counter))
                    engine.runAndWait()
                if angle > 120 and time.time() - last_feedback_time > 2 and last_feedback != "Elbow out":
                     workout_data['reps'][-1]['elbow_out'] = True
                     workout_data['feedback'].append("elbow out")
                     engine.say("Elbow out, pull it inwards")
                     engine.runAndWait()
                     last_feedback_time = time.time()
                     last_feedback = "Elbow out"
                elif angle <= 120 and last_feedback == "Elbow out":
                   last_feedback = ""
            except:
                pass
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        draw_counter += 1
        if draw_counter % 2 ==0:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()