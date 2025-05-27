import pandas as pd
import numpy as np
import time


start = time.time()
pd_data = pd.read_csv("data.txt", sep=";", na_values="?")
end = time.time()
print("Час зчитування pandas в секундах -", end - start)


start = time.time()
types = [("Date", "U10"), ("Time", "U8"), ("Global_active_power",
"float64"), ("Global_reactive_power", "float64"), ("Voltage",
"float64"), ("Global_intensity", "float64"), ("Sub_metering_1",
"float64"), ("Sub_metering_2", "float64"), ("Sub_metering_3",
"float64")]
np_data = np.genfromtxt("data.txt", missing_values=["?", np.nan],
delimiter=';', dtype=types, encoding="UTF=8", names=True)
end = time.time()
print("Час зчитування numpy в секундах -", end - start)

start = time.time()
pd_task1 = pd_data[pd_data["Global_active_power"] > 5]
print(pd_task1)
end = time.time()
print("Час виконання 1 завдання з pandas", end - start)

start = time.time()
np_task1 = np_data[np_data["Global_active_power"] > 5]
print(np_task1)
end = time.time()
print("Час виконання 1 завдання з numpy", end - start)

start = time.time()
pd_task2 = pd_data[pd_data["Voltage"] > 235]
print(pd_task2)
end = time.time()
print("Час виконання 2 завдання з pandas", end - start)

start = time.time()
np_task2 = np_data[np_data["Voltage"] > 235]
print(np_task2)
end = time.time()
print("Час виконання 2 завдання з numpy", end - start)

start = time.time()
pd_task3 = pd_data[(pd_data["Global_intensity"] >= 19) & (pd_data["Global_intensity"] <= 20) &
                   (pd_data["Sub_metering_2"] > pd_data["Sub_metering_3"])]
print(pd_task3)
end = time.time()
print("Час виконання 3 завдання з pandas", end - start)

start = time.time()
np_task3 = np_data[(np_data["Global_intensity"] >= 19) & (np_data["Global_intensity"] <= 20) &
                   (np_data["Sub_metering_2"] > np_data["Sub_metering_3"])]
print(np_task3)
end = time.time()
print("Час виконання 3 завдання з numpy", end - start)

start = time.time()
pd_task4 = pd_data.sample(n=500000)
meanSub1 = pd_task4[["Sub_metering_1"]].mean()
meanSub2 = pd_task4[["Sub_metering_2"]].mean()
meanSub3 = pd_task4[["Sub_metering_3"]].mean()
print(pd_task4)
print(meanSub1)
print(meanSub2)
print(meanSub3)
end = time.time()
print("Час виконання 4 завдання з pandas", end - start)

start = time.time()
i = np.random.choice(len(np_data), size=500000, replace=False)
np_task4 = np_data[i]
meanSub1 = [np.nanmean(np_task4["Sub_metering_1"])]
meanSub2 = [np.nanmean(np_task4["Sub_metering_2"])]
meanSub3 = [np.nanmean(np_task4["Sub_metering_3"])]
print(meanSub1)
print(meanSub2)
print(meanSub3)
end = time.time()
print("Час виконання 4 завдання з numpy", end - start)

start = time.time()
pd_task5 = pd_data[(pd_data["Time"] > "18:00:00") &
                   (pd_data["Global_active_power"] > 6) &
                   (pd_data["Sub_metering_2"] > pd_data["Sub_metering_1"]) &
                   (pd_data["Sub_metering_2"] > pd_data["Sub_metering_3"])]

length = len(pd_task5)

first_half = pd_task5.iloc[:length // 2]
second_half = pd_task5.iloc[length // 2:]

first_half_selected = first_half.iloc[::3]
second_half_selected = second_half.iloc[::4]

result = pd.concat([first_half_selected, second_half_selected]).reset_index()
print(result)
end = time.time()
print("Час виконання 5 завдання з pandas", end - start)

start = time.time()
np_task5 = np_data[(np_data["Time"] > "18:00:00") &
                   (np_data["Global_active_power"] > 6) &
                   (np_data["Sub_metering_2"] > np_data["Sub_metering_1"]) &
                   (np_data["Sub_metering_2"] > np_data["Sub_metering_3"])]
length = len(np_task5)

first_half = pd_task5.iloc[:length // 2]
second_half = pd_task5.iloc[length // 2:]

first_half_selected = first_half.iloc[::3]
second_half_selected = second_half.iloc[::4]

result = np.concatenate((first_half_selected, second_half_selected))
print(result)
end = time.time()
print("Час виконання 5 завдання з numpy", end - start)