# Cricket AI Match Predictor

## 📌 Project Overview
This project is an AI-powered cricket match predictor that estimates the **run rate and possible final score** based on match conditions. It uses **machine learning (Random Forest)** to analyze match details and predict future scores.

## 🚀 Features
- **Predicts Run Rate**: AI estimates the current and future run rate based on match conditions.
- **Visualizes Score Projection**: Generates a graph showing possible scores for the remaining overs.
- **User-Friendly Interface**: Uses Gradio for a simple web-based UI.
- **Interactive Inputs**: Users can enter match details like overs, wickets, and opponent strength.
- **Real-time AI Calculation**: Uses trained machine learning models to make predictions instantly.

## 🏗️ How It Works
1. **Input Match Data**
   - Overs played
   - Wickets fallen
   - Innings (1st or 2nd)
   - Current score
   - Opponent strength
   - Pitch size
   - Stadium location

2. **Machine Learning Model**
   - The model is trained on cricket data using **Random Forest Regressor**.
   - It analyzes past match trends and predicts a likely run rate.

3. **Prediction & Graph**
   - AI predicts the **run rate**.
   - A **graph is generated** showing how the score may evolve in the remaining overs.

## 📦 Installation
Make sure you have **Python** installed, then follow these steps:

```sh
pip install gradio pandas numpy scikit-learn tensorflow torch matplotlib seaborn
```

## ▶️ Running the Project
```sh
python app.py
```
This will open a **Gradio web UI** in your browser where you can enter match details and see predictions.

## 🛠️ Tech Stack
- **Python**: Core programming language
- **Gradio**: For web interface
- **Machine Learning**: Random Forest Regressor
- **Matplotlib & Seaborn**: For data visualization
- **NumPy & Pandas**: For data handling

## 📊 Sample Data Example
| Overs | Wickets | Innings | Score | Opponent Strength | Pitch Size | Stadium Location |
|-------|---------|---------|-------|------------------|-----------|----------------|
| 20    | 3       | 1       | 120   | 5                | 70        | 10             |

Predicted Run Rate: **7.5 runs per over**

## 🏆 Future Improvements
- Integrate **live match data** from APIs.
- Improve model accuracy using **deep learning**.
- Deploy as a **mobile app** for real-time cricket analysis.

## 📜 License
This project is open-source under the **MIT License**.

## 🤝 Contribution
Want to improve this project? Feel free to **fork and submit a pull request!**

---
Enjoy predicting cricket matches with AI! 🏏🚀



![Screenshot 2025-03-08 213730](https://github.com/user-attachments/assets/9958196a-7f14-4a97-a5ce-e87dc2ff277b)
![Screenshot 2025-03-08 213738](https://github.com/user-attachments/assets/c5a7a60e-79f7-4001-be45-a0d90f80f310)
![Screenshot 2025-03-08 214904](https://github.com/user-attachments/assets/81bb9cfb-6fb7-43a8-b927-a3a272fc30c6)
![Screenshot 2025-03-08 214915](https://github.com/user-attachments/assets/9face1cc-0fcb-4424-9175-1e352e9b62b4)
![Screenshot 2025-03-08 214925](https://github.com/user-attachments/assets/2fe2f597-7cc3-48fd-aa0a-a18ca04180c0)



