#test case exp no:2
import pandas as pd
import numpy as np
true_labels = ["T-shirt", "Trouser", "Pullover", "Sneaker"]
predicted_labels = ["T-shirt", "Dress", "Pullover", "Ankle boot"]
df = pd.DataFrame({
    "Input Image": true_labels,
    "True Label": true_labels,
    "Predicted Label": predicted_labels
})
df["Correct (Y/N)"] = np.where(df["True Label"] == df["Predicted Label"], "Y", "N")
print(df)

output:
<img width="489" height="111" alt="Screenshot 2025-08-13 115418" src="https://github.com/user-attachments/assets/4f7a832e-30cf-4002-a7b2-ba2127ae537e" />


#test case exp no:2
import pandas as pd
import numpy as np
data = [
    ["Image of 7", 7, 7],
    ["Image of 3", 3, 3],
    ["Image of 8", 8, 8],
    ["Image of 1", 1, 2]
]
df = pd.DataFrame(data, columns=["Input Digit Image", "Expected Label", "Model Output"])
df["Correct (Y/N)"] = np.where(df["Expected Label"] == df["Model Output"], "Y", "N")
print(df)

output:
<img width="619" height="103" alt="Screenshot 2025-08-13 115526" src="https://github.com/user-attachments/assets/c8246037-e983-490a-a343-b1e8604cdcea" />
