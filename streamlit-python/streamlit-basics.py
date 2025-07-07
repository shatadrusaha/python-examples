"""                     Import libraries.                       """
import streamlit as st
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



