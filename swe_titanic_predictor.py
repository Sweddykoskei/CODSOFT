import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
titanic_data = pd.read_csv(r'C:\Users\USER\OneDrive\Desktop\Titanic-Dataset.csv')

# Feature Engineering
titanic_data['Title'] = titanic_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1
titanic_data = titanic_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Split the data into features and target
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Initialize models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42)
}

# Function to train and evaluate a model
def train_and_evaluate(model_name):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', models[model_name])])
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    return model, cv_scores.mean()

# Function to make a prediction
def make_prediction(passenger_id, model):
    passenger_data = titanic_data.loc[titanic_data['PassengerId'] == passenger_id]
    if not passenger_data.empty:
        actual_survival = 'Survived' if passenger_data['Survived'].values[0] == 1 else 'Did not survive'
        passenger_data = passenger_data.drop('Survived', axis=1)
        prediction = model.predict(passenger_data)[0]
        return actual_survival, 'Survived' if prediction == 1 else 'Did not survive'
    else:
        return 'Unknown', 'No passenger found with the given PassengerId.'

# Train the default model and get the initial CV score
loaded_model, cv_score = train_and_evaluate('RandomForest')

# GUI function to display the dataset and get PassengerId input for prediction
def gui_display_dataset():
    root = tk.Tk()
    root.title("Titanic Dataset")

    # Frame for TreeView
    frame1 = tk.LabelFrame(root, text="Passenger Data")
    frame1.pack(fill="both", expand="yes", padx=20, pady=10)

    # Treeview Widget
    tv1 = ttk.Treeview(frame1)
    tv1.pack()

    # Define our columns
    tv1['columns'] = ('PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone')

    # Format our columns
    tv1.column("#0", width=0, stretch=tk.NO)
    for col in tv1['columns']:
        tv1.column(col, anchor=tk.CENTER, width=80)
        tv1.heading(col, text=col, anchor=tk.CENTER)

    # Add data to the treeview
    for index, row in titanic_data.iterrows():
        tv1.insert("", tk.END, iid=index, values=list(row))

    # Dropdown menu for model selection
    model_var = tk.StringVar(root)
    model_choices = list(models.keys())
    model_var.set(model_choices[0])  # set the default option
    model_menu = ttk.OptionMenu(root, model_var, model_choices[0], *model_choices)
    model_menu.pack()

    # Reload Button
    def reload_data():
        selected_model = model_var.get()
        global loaded_model, cv_score
        loaded_model, cv_score = train_and_evaluate(selected_model)
        messagebox.showinfo("Model Reloaded", f"Model: {selected_model}\nCV Score: {cv_score:.4f}")

    reload_button = ttk.Button(root, text="Reload Model", command=reload_data)
    reload_button.pack()

    # Label to show the survival status
    survival_label = tk.Label(root, text="", font=("Helvetica", 16))
    survival_label.pack(pady=20)

    # Select Button
    def select_record():
        selected = tv1.focus()
        values = tv1.item(selected, 'values')
        passenger_id = values[0]
        survived, prediction = make_prediction(int(passenger_id), loaded_model)
        survival_label.config(text=f"PassengerId: {passenger_id}, Survived: {survived}")
        messagebox.showinfo("Prediction", f"PassengerId: {passenger_id}, Prediction: {prediction}")

    select_button = tk.Button(root, text="Select Record", command=select_record)
    select_button.pack(pady=10)

    # Function to visualize model performance
    def visualize_performance():
        fig, ax = plt.subplots()
        ax.barh(['Model'], [cv_score], color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Cross-Validation Score')
        ax.set_title('Model Performance')
        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Button to visualize model performance
    visualize_button = tk.Button(root, text="Visualize Performance", command=visualize_performance)
    visualize_button.pack(pady=10)

    # Run the application
    root.mainloop()

# Call the GUI function
gui_display_dataset()
