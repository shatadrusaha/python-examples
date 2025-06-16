from IPython.display import Markdown, display
import re


def generate_table_of_contents():
    # Open the notebook file
    with open("explore-polars.ipynb", "r") as f:
        notebook_content = f.read()

    # Extract Markdown headers
    headers = re.findall(r"#+\s+.*", notebook_content)

    # Generate Table of Contents
    toc = []
    for header in headers:
        level = header.count("#")  # Determine the header level
        title = header.strip("#").strip()  # Extract the title
        link = (
            re.sub(r"[^\w\s]", "", title).replace(" ", "-").lower()
        )  # Generate a link
        toc.append(f"{'  ' * (level - 1)}- [{title}](#{link})")

    # Display the Table of Contents
    display(Markdown("\n".join(toc)))


# Call the function to generate the Table of Contents
generate_table_of_contents()