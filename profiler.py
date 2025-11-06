def calculateNutritionalTarget(weight, height, age, gender, activeLevel, goal, sugarLevel):
    if gender.lower() == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    activityMultipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725}
    tdee = bmr * activityMultipliers.get(activeLevel.lower(), 1.55)

    totalCalories = tdee - 400 if goal == 'lose' else tdee + 400 if goal == 'gain' else tdee

    if sugarLevel.lower() == 'high':
        totalProtein = (totalCalories * 0.40) / 4
        fat = (totalCalories * 0.35) / 9
        carbs = (totalCalories * 0.25) / 4
        sugar = 20
        fiber = 35
    else:
        totalProtein = (totalCalories * 0.35) / 4
        fat = (totalCalories * 0.30) / 9
        carbs = (totalCalories * 0.35) / 4
        sugar = 45
        fiber = 25

    return [
        round(totalCalories, 1),
        round(fat, 1),
        round(fat * 0.3, 1),
        300/1000,
        2300/1000,
        round(carbs, 1),
        round(fiber, 1),
        round(sugar, 1),
        round(totalProtein, 1)
    ]