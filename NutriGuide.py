import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Sample data (fruit and vegetable names, corresponding nutrition levels, uses, and health benefits)
items = ['apple', 'banana', 'orange', 'kiwi', 'grape', 'strawberry', 'pineapple', 'watermelon', 'carrot', 'broccoli', 
         'spinach', 'tomato', 'cucumber', 'bell pepper', 'lettuce', 'potato', 'sweet potato', 'onion', 'garlic', 
         'ginger', 'avocado', 'mango', 'peach', 'pear', 'blueberry', 'raspberry', 'blackberry', 'papaya', 'lemon', 
         'lime', 'celery', 'zucchini', 'asparagus', 'eggplant', 'corn', 'cauliflower', 'cabbage', 'radish', 'pumpkin', 
         'squash', 'green bean', 'beet', 'cherry', 'fig', 'plum', 'grapefruit', 'nectarine', 'apricot']

calories = [52, 89, 47, 61, 69, 33, 50, 30, 41, 34, 23, 18, 15, 31, 15, 77, 86, 40, 149, 80, 160, 60, 60, 57, 60, 52, 
            52, 43, 29, 29, 16, 33, 20, 25, 86, 25, 25, 16, 30, 26, 26, 47, 74, 47, 32, 44, 48, 48]

carbs = [14, 23, 12, 15, 18, 8, 13, 8, 10, 7, 4, 4, 3, 6, 3, 18, 20, 10, 33, 15, 9, 15, 15, 15, 15, 14, 12, 10, 9, 9, 
         3, 7, 4, 4, 19, 4, 4, 4, 5, 6, 6, 12, 18, 12, 8, 11, 12, 12]

fiber = [2.4, 2.6, 2.4, 3, 0.6, 2, 1.4, 0.4, 2.8, 2.6, 2.2, 1.2, 0.5, 1.7, 1.3, 3, 3, 1.7, 6.3, 2, 6.7, 1.6, 2.6, 3.1, 
         3.6, 6.5, 6.7, 2.7, 2.8, 2.8, 1.6, 1.6, 1.6, 1.6, 2.7, 1.6, 2.7, 1.6, 1.9, 2.5, 2.5, 3.3, 1.4, 2.6, 1.7, 1.6, 2.6, 2.6]

vitamin_c = [5, 10, 59, 93, 16.3, 58.8, 47.8, 8.1, 7.2, 89.2, 28.1, 13.7, 2.8, 119.8, 3.7, 19.7, 2.5, 7.4, 31.2, 5, 10, 36.4, 6.6, 12, 
             9.7, 26.2, 21, 60.9, 38.7, 29.1, 3.1, 17.0, 5.4, 2.2, 10.2, 48.2, 36.6, 14.8, 12, 8.1, 6.7, 16.3, 10.0, 13.5, 5.6, 8.1, 9.5, 10.0]

uses = ['Eating raw', 'Eating raw', 'Juice, Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 
        'Eating raw', 'Eating raw, Salad', 'Stir-fry, Salad', 'Salad, Smoothies', 'Salad, Cooking', 'Salad, Cooking', 
        'Salad, Cooking', 'Salad, Sandwich', 'Fries, Salad', 'Fries, Salad', 'Cooking', 'Cooking', 'Cooking, Tea', 
        'Salad, Guacamole', 'Eating raw, Salad, Smoothies', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 
        'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Cooking, Salad', 
        'Cooking, Salad', 'Cooking, Salad', 'Cooking, Salad', 'Cooking, Salad', 'Cooking, Salad', 'Cooking, Salad', 
        'Cooking, Salad', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 'Eating raw', 
        'Eating raw']

health_benefits = ['Rich in antioxidants', 'High in potassium', 'Rich in vitamin C', 'High in vitamin C and fiber', 
                   'High in antioxidants', 'Rich in vitamin C and antioxidants', 'Rich in vitamin C and manganese', 
                   'Hydrating, Low in calories', 'Rich in vitamin A and beta-carotene', 'Rich in fiber and vitamin C', 
                   'Rich in iron and vitamin K', 'Rich in lycopene', 'Hydrating, Low in calories', 'Rich in vitamin C and antioxidants', 
                   'Low in calories, Rich in water', 'Rich in vitamin C and potassium', 'Rich in vitamin C and potassium', 
                   'Rich in antioxidants', 'Rich in antioxidants', 'Rich in antioxidants', 'Rich in healthy fats', 
                   'Rich in vitamin C, A, and E', 'Rich in vitamin C', 'Rich in vitamin C', 'Rich in vitamin C', 
                   'Rich in antioxidants', 'Rich in antioxidants', 'Rich in antioxidants', 'Rich in vitamin C', 
                   'Rich in vitamin C', 'Rich in vitamin K', 'Rich in vitamin A', 'Rich in vitamin C', 'Rich in antioxidants', 
                   'Rich in vitamin B', 'Rich in antioxidants', 'Rich in vitamin C', 'Rich in vitamin C', 'Rich in vitamin C', 
                   'Rich in vitamin C', 'Rich in vitamin K', 'Rich in fiber', 'Rich in antioxidants', 'Rich in vitamin A', 
                   'Rich in vitamin C', 'Rich in vitamin C', 'Rich in vitamin C']

# Encode fruit and vegetable names using LabelEncoder
label_encoder = LabelEncoder()
encoded_items = label_encoder.fit_transform(items)

# Prepare features and target variables
X = np.array(encoded_items).reshape(-1, 1)
y_calories = np.array(calories)
y_carbs = np.array(carbs)
y_fiber = np.array(fiber)
y_vitamin_c = np.array(vitamin_c)

# Train separate regression models for each nutrition level
model_calories = RandomForestRegressor()
model_calories.fit(X, y_calories)

model_carbs = RandomForestRegressor()
model_carbs.fit(X, y_carbs)

model_fiber = RandomForestRegressor()
model_fiber.fit(X, y_fiber)

model_vitamin_c = RandomForestRegressor()
model_vitamin_c.fit(X, y_vitamin_c)

def predict_nutrition(item_name):
    encoded_item = label_encoder.transform([item_name])
    predicted_calories = model_calories.predict(encoded_item.reshape(-1, 1))[0]
    predicted_carbs = model_carbs.predict(encoded_item.reshape(-1, 1))[0]
    predicted_fiber = model_fiber.predict(encoded_item.reshape(-1, 1))[0]
    predicted_vitamin_c = model_vitamin_c.predict(encoded_item.reshape(-1, 1))[0]

    response = f"Predicted nutrition levels for {item_name}:\n"
    response += f"Calories: {predicted_calories:.2f} kcal\n"
    response += f"Carbohydrates: {predicted_carbs:.2f} g\n"
    response += f"Fiber: {predicted_fiber:.2f} g\n"
    response += f"Vitamin C: {predicted_vitamin_c:.2f} mg\n\n"
    response += f"Uses: {uses[items.index(item_name)]}\n"
    response += f"Health Benefits: {health_benefits[items.index(item_name)]}\n"

    return response

# Main chatbot function
def chatbot():
    print("Welcome to the Fruit and Vegetable Nutrition Chatbot!")
    print("Enter the name of a fruit or vegetable to predict its nutrition levels and learn more about it.")
    print("Type 'exit' to end the chat.")

    while True:
        item_name = input("Enter the name of a fruit or vegetable: ").lower()
        if item_name == "exit":
            print("Thank you for using the Fruit and Vegetable Nutrition Chatbot. Goodbye!")
            break

        if item_name in items:
            response = predict_nutrition(item_name)
            print(response)
        else:
            print("Sorry, I couldn't find information about that item. Please try again.")

# Run the chatbot
if __name__ == "__main__":
    chatbot()


