import plotly.express as px
import plotly.io as pio

# Set renderer to open the plot in a browser
pio.renderers.default = "browser"  # OR use "vscode" for VS Code

# Sample dataset
df = px.data.iris()

# Create a scatter plot
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 title="Plotly Test Visualization")

# Show the plot
fig.show()
 
#%%
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Sine Wave", color="blue")

# Add labels and title
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Matplotlib Visualization in VS Code")
plt.legend()
plt.grid()

# Show the plot (for VS Code and .py scripts)
plt.show()

# %%
