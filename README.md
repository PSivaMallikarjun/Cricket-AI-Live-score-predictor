# Cricket-AI-Live-score-predictor
"The Projected Score (how many runs the team might score in the remaining overs), The Run Rate (how fast a team is scoring runs)."
What Does This Code Do?
This is a Cricket AI Chatbot that helps predict:

The Run Rate (how fast a team is scoring runs).
The Projected Score (how many runs the team might score in the remaining overs).
It does this based on some match conditions, like:

How many overs have been played
How many wickets are left
If it's the 1st or 2nd innings
Current score
Strength of the opponent
Pitch size and stadium location
Breaking It Down Step by Step
1. Installing Required Packages
The first line installs all the necessary tools (Gradio, Pandas, NumPy, etc.) for building this AI chatbot.

python

!pip install gradio pandas numpy scikit-learn tensorflow torch matplotlib seaborn
Gradio → Creates a simple user interface (UI).
Pandas → Handles data.
NumPy → Works with numbers and arrays.
Scikit-learn → Helps with machine learning.
Matplotlib & Seaborn → Used for creating graphs.
TensorFlow & Torch → AI and deep learning tools (not used directly here).
2. Creating Sample Data (Simulating a Cricket Match)
python

data = pd.DataFrame({
    'Overs': np.random.randint(5, 50, 500),
    'Wickets': np.random.randint(0, 10, 500),
    'Innings': np.random.choice([1, 2], 500),
    'Score': np.random.randint(30, 400, 500),
    'Opponent': np.random.randint(1, 10, 500),
    'Pitch_Size': np.random.randint(50, 80, 500),
    'Stadium_Location': np.random.randint(1, 20, 500),
    'Run_Rate': np.random.uniform(3.0, 10.0, 500)
})
This generates 500 random cricket match scenarios.
Each match has different values for overs, wickets, innings, score, opponent strength, pitch size, and stadium location.
3. Splitting the Data for Training
python

y = data['Run_Rate']
X = data.drop(columns=['Run_Rate'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Here, we separate the "Run Rate" column (which we want to predict) from other columns.
Then we split the data into training (80%) and testing (20%) to train the AI model.
4. Training the AI Model
python

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
We use Random Forest Regressor, a machine learning model.
The model learns from the training data to predict future run rates based on input conditions.
5. Function to Predict Run Rate
python

def predict_run_rate(overs, wickets, innings, score, opponent, pitch_size, stadium_location):
    input_data = np.array([[overs, wickets, innings, score, opponent, pitch_size, stadium_location]])
    predicted_run_rate = rf_model.predict(input_data)[0]
    return f'Predicted Run Rate: {predicted_run_rate:.2f}'
This function takes match details (overs played, wickets left, score, etc.).
It predicts the expected run rate using the trained model.
6. Function to Generate a Graph
python
def visualize_predictions(overs, wickets, innings, score, opponent, pitch_size, stadium_location):
    balls_left = (50 - overs) * 6
    predicted_run_rate = rf_model.predict(np.array([[overs, wickets, innings, score, opponent, pitch_size, stadium_location]]))[0]
    predicted_scores = [score + (predicted_run_rate * i) for i in range(0, balls_left // 6 + 1)]
    overs_remaining = list(range(overs, overs + len(predicted_scores)))
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=overs_remaining, y=predicted_scores, marker='o', linestyle='dashed', color='b')
    plt.xlabel('Overs Bowled')
    plt.ylabel('Predicted Score')
    plt.title('Projected Score Over Remaining Overs')
    plt.grid(True)
    
    plt.savefig("prediction_graph.png")
    return "prediction_graph.png"
This function creates a graph to show:
How the score might increase in the remaining overs.
A visual trend based on predicted run rate.
7. Creating the Gradio User Interface
python
demo = gr.Interface(
    fn=[predict_run_rate, visualize_predictions],
    inputs=[
        gr.Number(label='Overs'),
        gr.Number(label='Wickets Left'),
        gr.Number(label='1st or 2nd Innings (1/2)'),
        gr.Number(label='Current Score'),
        gr.Number(label='Opponent Strength (1-10)'),
        gr.Number(label='Pitch Size (50-80 meters)'),
        gr.Number(label='Stadium Location (1-20)')
    ],
    outputs=[gr.Textbox(label='Predicted Run Rate'), gr.Image(label='Prediction Graph')],
    title='Cricket AI Chatbot',
    description='Enter match details to get AI-powered match predictions!'
)
Gradio creates a web-based UI for users to enter match details.
The chatbot returns:
A predicted run rate (as text).
A graph showing the expected score progression.
8. Running the Web App
python
demo.launch()
This launches the chatbot in a web browser.

![Screenshot 2025-03-08 213730](https://github.com/user-attachments/assets/9958196a-7f14-4a97-a5ce-e87dc2ff277b)
![Screenshot 2025-03-08 213738](https://github.com/user-attachments/assets/c5a7a60e-79f7-4001-be45-a0d90f80f310)
![Screenshot 2025-03-08 214904](https://github.com/user-attachments/assets/81bb9cfb-6fb7-43a8-b927-a3a272fc30c6)
![Screenshot 2025-03-08 214915](https://github.com/user-attachments/assets/9face1cc-0fcb-4424-9175-1e352e9b62b4)
![Screenshot 2025-03-08 214925](https://github.com/user-attachments/assets/2fe2f597-7cc3-48fd-aa0a-a18ca04180c0)



