import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'name':['rose','lily','lotus','cactus','jasmine','Bamboo','Tulip'],
    'sunlight':[12,8,11,8,18,15,21],
    'plant-height':[110,140,80,210,160,500,200]
})

plt.xlabel('Sunlight')
plt.ylabel('Height of plant')
plt.title('Relationship between sunlight and height of the plant')
plt.scatter(df['sunlight'],df['plant-height'],marker='*')
plt.show()

correlation = df['sunlight'].corr(df['plant-height'])
print(f"The correlation between sunlight and plant height is {correlation}")


if correlation < 0:
    sign = "negative"
elif correlation > 0:
    sign = "positive"
else:
    sign = "neither"
print(f"The correlation coefficient is {sign}.")

strength = "strong" if abs(correlation) > 0.5 else "weak"
print(f"The correlation is {strength}.")
threshold = 0.7
if(abs(correlation)>= threshold):
    print(f"There is significant association between plant-growth rate and sunlight")
else:
    print("There is no significant association between plant-growth rate and sunlight")
