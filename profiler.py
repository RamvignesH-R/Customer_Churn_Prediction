def calculateNutritionalTarget(weight,height,age,gender,activeLevel,goal,sugarLevel):

    #Harris-Benedict Equation
    if gender.lower()=='male':
        bmr=88.362+(13.397*weight)+(4.799*height)-(5.677*age)
    else:
        bmr=447.593+(9.247*weight)+(3.098*height)-(4.330*age)

    #Total Daily Energy Expenditure
    activityMultipliers={'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725}
    tdee=bmr*activityMultipliers.get(activeLevel, 1.55)

    if goal=='lose':
        totalCalories=tdee-400
    elif goal=='gain':
        totalCalories=tdee+400
    else:
        totalCalories=tdee

    if sugarLevel=='high':
        totalProtein=(totalCalories*0.40)/4 
        fat=(totalCalories*0.35)/9
        carbs=(totalCalories*0.25)/4
        sugar=20
        fiber=35
    else:
        totalProtein=(totalCalories*0.35)/4
        fat=(totalCalories*0.30)/9
        carbs=(totalCalories*0.35)/4
        sugar=45
        fiber=25

    
    targetValues=[
        totalCalories,         
        fat,              
        fat * 0.3,        
        300,                     
        2300,                    
        carbs,            
        fiber,            
        sugar,            
        totalProtein          
    ]

    return targetValues

