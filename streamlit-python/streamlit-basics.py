"""                     Import libraries.                       """
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from PIL import Image

# Display a title and some text in the Streamlit app.
st.title("This is title of the Streamlit app")
st.write("Hello world!")

# Display a header and some text.
st.header("This is a header")
st.write("This is some text under the header.")

# Create two buttons and print their states.
pressed_first = st.button("1st button, Press me!")
print(f"'pressed_first': {pressed_first}")

pressed_second = st.button("2nd button, Press me!")
print(f"'pressed_second': {pressed_second}\n")

# Display a subheader and some text.
st.subheader("This is a subheader")
st.write("This is some text under the subheader.")

# Display a markdown text.
st.markdown("This is a **markdown** text with *italic* and **bold** formatting.")

#  Display a caption.
st.caption("This is a caption for the Streamlit app.")

# Display a code block.
code_example = """
def hello_world():
    print("Hello, world!")
"""
st.code(code_example, language='python')

# Display a divider.
st.divider()

# Display an image.
path_image = './images/streamlit-logo-primary-colormark-darktext-2458358719.png' # Path to the image file.
image = Image.open(fp=path_image) # Open the image file using PIL.
st.image(image=image, caption='Streamlit Logo #1', use_container_width=True) # Display the image with a caption.

# Display another image.
path_image = './images/streamlit_hero-2314562563.jpeg'
st.image(image=path_image, caption='Streamlit Logo #2', use_container_width=True)

# Display a dataframe.
st.subheader("DataFrame Example")
df = pd.DataFrame({
    'Column 1': [1, 2, 3],
    'Column 2': ['A', 'B', 'C'],
    'Column 3': [True, False, True]
})
st.dataframe(data=df, use_container_width=True)

# Display an editable dataframe.
st.subheader("Editable DataFrame Example")
df_editable = st.data_editor(data=df, use_container_width=True)
# print(df_editable)

# Display a table (static).
st.subheader("Table (static) Example")
st.table(data=df)

# Display metrics.
st.subheader("Metrics Example")
st.metric(label="Temperature", value="20 °C", delta="1 °C")

# Display json data.
st.subheader("JSON Example")
json_data = {
    "name": "Streamlit",
    "version": "1.0",
    "features": ["easy to use", "interactive", "fast"],
    "active": True
}
st.json(body=json_data)

st.write("Disctionary view of the JSON data:", json_data)

# Display an area chart.
st.subheader("Area Chart Example")
df_chart = pd.DataFrame(
    data=np.random.randn(100, 3),
    columns=['A', 'B', 'C']
)
st.area_chart(data=df_chart, use_container_width=True)

# Display a bar chart.
st.subheader("Bar Chart Example")
st.bar_chart(data=df_chart, use_container_width=True)

# Display a line chart.
st.subheader("Line Chart Example")
st.line_chart(data=df_chart, use_container_width=True)

# Display scatter plot.
st.subheader("Scatter Plot Example")
st.scatter_chart(data=df_chart, x='A', y='B', use_container_width=True)

# Display a map.
st.subheader("Map Example")
map_data = pd.DataFrame(
    data=np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)
st.map(data=map_data, use_container_width=True)

# Display a pyplot chart.
st.subheader("Pyplot Chart Example")
fig, ax = plt.subplots()
ax.plot(df_chart['A'], label='A')
ax.plot(df_chart['B'], label='B')
ax.plot(df_chart['C'], label='C')
ax.set_title('Pyplot Line Chart')
ax.set_xlabel('Index')
ax.set_ylabel('Values')
ax.legend()
st.pyplot(fig=fig, use_container_width=True)

# Display a form.
st.subheader("Form Example")
with st.form(key='my_form'):
    name = st.text_input(label='Name')
    age = st.number_input(label='Age', min_value=0, max_value=100)
    dob = st.date_input(label='Date of Birth', value=dt.today())
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        st.write(f"Name: {name}, Age: {age}")
        st.success("Form submitted successfully!")

# Advanced form with multiple inputs.
st.subheader("Advanced Form Example")
form_values = {
    'name': None,
    'age': None,
    'dob': None,
    'email': None
}
date_min = dt(1900, 1, 1) # Minimum date for date input.
date_max = dt.today() # Maximum date for date input.
with st.form(key='advanced_form'):
    form_values['name'] = st.text_input(label='Name')
    form_values['age'] = st.number_input(label='Age', min_value=0, max_value=100)
    form_values['email'] = st.text_input(label='Email')
    form_values['dob'] = st.date_input(
        label='Date of Birth', 
        min_value=date_min, 
        max_value=date_max
    )
    advanced_submit_button = st.form_submit_button(label='Submit')
    if advanced_submit_button:
        if not all(form_values.values()):
            st.error("Please fill in all fields.")
        else:
            st.balloons()
            st.write(f"Name: {form_values['name']}, Age: {form_values['age']}, Email: {form_values['email']}")
            st.success("Advanced form submitted successfully!")
