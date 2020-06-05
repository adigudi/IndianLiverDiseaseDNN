import pandas as pd
df = pd.read_csv('indian_liver_patient.csv')
df.to_csv('output.csv', index=False)
