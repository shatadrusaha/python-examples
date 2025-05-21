{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Connected to .venv (Python 3.11.10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd65164d-edbe-4c19-9e3b-f75acad00ddd",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'explore-polars.ipynb'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 24\u001b[39m\n\u001b[32m     21\u001b[39m     display(Markdown(\u001b[33m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[33m\"\u001b[39m.join(toc)))\n\u001b[32m     23\u001b[39m \u001b[38;5;66;03m# Call the function to generate the Table of Contents\u001b[39;00m\n\u001b[32m---> \u001b[39m\u001b[32m24\u001b[39m \u001b[43mgenerate_table_of_contents\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 6\u001b[39m, in \u001b[36mgenerate_table_of_contents\u001b[39m\u001b[34m()\u001b[39m\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34mgenerate_table_of_contents\u001b[39m():\n\u001b[32m      5\u001b[39m     \u001b[38;5;66;03m# Open the notebook file\u001b[39;00m\n\u001b[32m----> \u001b[39m\u001b[32m6\u001b[39m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mexplore-polars.ipynb\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mr\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[32m      7\u001b[39m         notebook_content = f.read()\n\u001b[32m      9\u001b[39m     \u001b[38;5;66;03m# Extract Markdown headers\u001b[39;00m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/Study/github/python-examples/polars-exploration/.venv/lib/python3.11/site-packages/IPython/core/interactiveshell.py:326\u001b[39m, in \u001b[36m_modified_open\u001b[39m\u001b[34m(file, *args, **kwargs)\u001b[39m\n\u001b[32m    319\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m file \u001b[38;5;129;01min\u001b[39;00m {\u001b[32m0\u001b[39m, \u001b[32m1\u001b[39m, \u001b[32m2\u001b[39m}:\n\u001b[32m    320\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[32m    321\u001b[39m         \u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33mIPython won\u001b[39m\u001b[33m'\u001b[39m\u001b[33mt let you open fd=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfile\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m by default \u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m    322\u001b[39m         \u001b[33m\"\u001b[39m\u001b[33mas it is likely to crash IPython. If you know what you are doing, \u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m    323\u001b[39m         \u001b[33m\"\u001b[39m\u001b[33myou can use builtins\u001b[39m\u001b[33m'\u001b[39m\u001b[33m open.\u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m    324\u001b[39m     )\n\u001b[32m--> \u001b[39m\u001b[32m326\u001b[39m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mio_open\u001b[49m\u001b[43m(\u001b[49m\u001b[43mfile\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m*\u001b[49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m*\u001b[49m\u001b[43m*\u001b[49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[31mFileNotFoundError\u001b[39m: [Errno 2] No such file or directory: 'explore-polars.ipynb'"
     ]
    }
   ],
   "source": [
    "from IPython.display import Markdown, display\n",
    "import re\n",
    "\n",
    "\n",
    "def generate_table_of_contents():\n",
    "    # Open the notebook file\n",
    "    with open(\"explore-polars.ipynb\", \"r\") as f:\n",
    "        notebook_content = f.read()\n",
    "\n",
    "    # Extract Markdown headers\n",
    "    headers = re.findall(r\"#+\\s+.*\", notebook_content)\n",
    "\n",
    "    # Generate Table of Contents\n",
    "    toc = []\n",
    "    for header in headers:\n",
    "        level = header.count(\"#\")  # Determine the header level\n",
    "        title = header.strip(\"#\").strip()  # Extract the title\n",
    "        link = (\n",
    "            re.sub(r\"[^\\w\\s]\", \"\", title).replace(\" \", \"-\").lower()\n",
    "        )  # Generate a link\n",
    "        toc.append(f\"{'  ' * (level - 1)}- [{title}](#{link})\")\n",
    "\n",
    "    # Display the Table of Contents\n",
    "    display(Markdown(\"\\n\".join(toc)))\n",
    "\n",
    "\n",
    "# Call the function to generate the Table of Contents\n",
    "generate_table_of_contents()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'/Users/shaz/Study/github/python-examples'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.getcwd()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10238b26-8da4-4527-afe5-2a84934f5fa1",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "generate_table_of_contents() missing 1 required positional argument: 'filename'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mTypeError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[4]\u001b[39m\u001b[32m, line 28\u001b[39m\n\u001b[32m     26\u001b[39m example_file = \u001b[33m'\u001b[39m\u001b[33mexplore-polars.ipynb\u001b[39m\u001b[33m'\u001b[39m\n\u001b[32m     27\u001b[39m filename = os.path.join(os.getcwd(), example_folder, example_file)\n\u001b[32m---> \u001b[39m\u001b[32m28\u001b[39m \u001b[43mgenerate_table_of_contents\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[31mTypeError\u001b[39m: generate_table_of_contents() missing 1 required positional argument: 'filename'"
     ]
    }
   ],
   "source": [
    "from IPython.display import Markdown, display\n",
    "import re\n",
    "import os\n",
    "\n",
    "\n",
    "def generate_table_of_contents(filename):\n",
    "    # Open the notebook file\n",
    "    with open(filename, \"r\") as f:\n",
    "        notebook_content = f.read()\n",
    "\n",
    "    # Extract Markdown headers\n",
    "    headers = re.findall(r\"#+\\s+.*\", notebook_content)\n",
    "\n",
    "    # Generate Table of Contents\n",
    "    toc = []\n",
    "    for header in headers:\n",
    "        level = header.count(\"#\")  # Determine the header level\n",
    "        title = header.strip(\"#\").strip()  # Extract the title\n",
    "        link = (\n",
    "            re.sub(r\"[^\\w\\s]\", \"\", title).replace(\" \", \"-\").lower()\n",
    "        )  # Generate a link\n",
    "        toc.append(f\"{'  ' * (level - 1)}- [{title}](#{link})\")\n",
    "\n",
    "    # Display the Table of Contents\n",
    "    display(Markdown(\"\\n\".join(toc)))\n",
    "\n",
    "\n",
    "# Call the function to generate the Table of Contents\n",
    "example_folder = \"polars-exploration\"\n",
    "example_file = \"explore-polars.ipynb\"\n",
    "filename = os.path.join(os.getcwd(), example_folder, example_file)\n",
    "generate_table_of_contents()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c40b531-d37e-453e-bd53-fec33c77c849",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "  - [Explore [_'Polars'_](https://pola.rs)\\n\",](#explore-_polars_httpspolarsn)\n",
       "    - [Install\\n\",](#installn)\n",
       "    - [Load libraries\"](#load-libraries)\n",
       "    - [Load/Download (large) the dataset\\n\",](#loaddownload-large-the-datasetn)\n",
       "- [Download latest version of the dataset.\\n\",](#download-latest-version-of-the-datasetn)\n",
       "- [Get the 'csv' filename of the dataset.\\n\",](#get-the-csv-filename-of-the-datasetn)\n",
       "- [path_data = '/Users/shaz/.cache/kagglehub/datasets/saurabhbadole/latest-data-science-job-salaries-2024/versions/3'\\n\",](#path_data--usersshazcachekagglehubdatasetssaurabhbadolelatestdatasciencejobsalaries2024versions3n)\n",
       "- [Load the dataset into a Polars DataFrame.\\n\",](#load-the-dataset-into-a-polars-dataframen)\n",
       "- [Display the first few rows of the DataFrame.\\n\",](#display-the-first-few-rows-of-the-dataframen)\n",
       "- [display(df_ev.head())\\n\",](#displaydf_evheadn)\n",
       "    - [(OPTIONAL) Save and load the data in _'parquet'_ format locally\\n\",](#optional-save-and-load-the-data-in-_parquet_-format-locallyn)\n",
       "- [Define filename for 'parquet' file.\\n\",](#define-filename-for-parquet-filen)\n",
       "- [Save the data.\\n\",](#save-the-datan)\n",
       "- [Read the data back from the 'parquet' file.\\n\",](#read-the-data-back-from-the-parquet-filen)\n",
       "- [Define filename for `lazy` method.\\n\",](#define-filename-for-lazy-methodn)\n",
       "- [Convert the DataFrame to a lazy DataFrame and save it as a Parquet file.\\n\",](#convert-the-dataframe-to-a-lazy-dataframe-and-save-it-as-a-parquet-filen)\n",
       "- [Read the lazy DataFrame from the Parquet file.\\n\",](#read-the-lazy-dataframe-from-the-parquet-filen)\n",
       "- [Convert the lazy DataFrame to a regular DataFrame.\\n\",](#convert-the-lazy-dataframe-to-a-regular-dataframen)\n",
       "- [Display the first few rows of the DataFrame.\\n\",](#display-the-first-few-rows-of-the-dataframen)\n",
       "      - [[Inspecting the `DataFrame`](https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#inspecting-a-dataframe)\\n\",](#inspecting-the-dataframehttpsdocspolarsuserguideconceptsdatatypesandstructuresinspectingadataframen)\n",
       "      - [`head`\\n\",](#headn)\n",
       "      - [`tail`\\n\",](#tailn)\n",
       "      - [`describe`\\n\",](#describen)\n",
       "      - [`glimpse`\\n\",](#glimpsen)\n",
       "      - [`sample`\\n\",](#samplen)\n",
       "      - [`schema`\\n\",](#scheman)\n",
       "- [Define the numeirc and categorical columns.\\n\",](#define-the-numeirc-and-categorical-columnsn)\n",
       "- [Define the new schema for the `DataFrame`.\\n\",](#define-the-new-schema-for-the-dataframen)\n",
       "- [Apply the new schema to the DataFrame.\\n\",](#apply-the-new-schema-to-the-dataframen)\n",
       "    - [[Expressions and contexts](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/)\\n\",](#expressions-and-contextshttpsdocspolarsuserguideconceptsexpressionsandcontextsn)\n",
       "      - [Expressions\\n\",](#expressionsn)\n",
       "- [[(col(\\\"weight\\\")) / (col(\\\"height\\\").pow([dyn int: 2]))]\\n\",](#colweight--colheightpowdyn-int-2n)\n",
       "      - [Contexts\\n\",](#contextsn)\n",
       "- [df_select = df_salary.select(\\n\",](#df_select--df_salaryselectn)\n",
       "- [[\\n\",](#n)\n",
       "- [pl.col('work_year'),\\n\",](#plcolwork_yearn)\n",
       "- [pl.col('salary_in_usd'),\\n\",](#plcolsalary_in_usdn)\n",
       "- []\\n\",](#n)\n",
       "- [)\"](#)\n",
       "      - [Contexts (cont.)\\n\",](#contexts-contn)\n",
       "      - [Contexts (cont.)\\n\",](#contexts-contn)\n",
       "- [df_filter = df_salary.filter(\\n\",](#df_filter--df_salaryfiltern)\n",
       "- [(pl.col('work_year') > 2020),\\n\",](#plcolwork_year--2020n)\n",
       "- [(pl.col('salary_in_usd') > 100000)\\n\",](#plcolsalary_in_usd--100000n)\n",
       "- [)\\n\",](#n)\n",
       "      - [Contexts (cont.)\\n\",](#contexts-contn)\n",
       "- [pl.col('work_year'),\\n\",](#plcolwork_yearn)\n",
       "- [pl.col('remote_ratio'),\\n\",](#plcolremote_ration)\n",
       "    - [[Advanced Expressions](https://docs.pola.rs/user-guide/expressions/)\\n\",](#advanced-expressionshttpsdocspolarsuserguideexpressionsn)\n",
       "      - [Basic Operations\\n\",](#basic-operationsn)\n",
       "      - [Basic Operations (cont.)\\n\",](#basic-operations-contn)\n",
       "      - [Basic Operations (cont.)\\n\",](#basic-operations-contn)\n",
       "- [Binary representation:\\n\",](#binary-representationn)\n",
       "- [Bitwise AND\\n\",](#bitwise-andn)\n",
       "- [Bitwise OR\\n\",](#bitwise-orn)\n",
       "      - [Basic Operations (cont.)\\n\",](#basic-operations-contn)\n",
       "  - [pl.col('employment_type').approx_n_unique().alias('approx_n_unique_emp_types'), # approx_n_unique() is not supported for Categorical type\\n\",](#plcolemployment_typeapprox_n_uniquealiasapprox_n_unique_emp_types--approx_n_unique-is-not-supported-for-categorical-typen)\n",
       "      - [Basic Operations (cont.)\\n\",](#basic-operations-contn)\n",
       "- [TODO: Yet to be documented.\"](#todo-yet-to-be-documented)\n",
       "    - [[Transformations](https://docs.pola.rs/user-guide/transformations/)\\n\",](#transformationshttpsdocspolarsuserguidetransformationsn)\n",
       "      - [Joins (`join`, `join_where`, `join_asof`)\\n\",](#joins-join-join_where-join_asofn)\n",
       "      - [Concatenation (`concat`)\\n\",](#concatenation-concatn)\n",
       "    - [Transformations (cont.)\\n\",](#transformations-contn)\n",
       "    - [Transformations (cont.)\\n\",](#transformations-contn)\n",
       "    - [Transformations (cont.)\\n\",](#transformations-contn)\n",
       "- [Assuming the dataframe loaded, say 'df', has a data/datetime column called 'Date'/'DateTime'.\\n\",](#assuming-the-dataframe-loaded-say-df-has-a-datadatetime-column-called-datedatetimen)"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from IPython.display import Markdown, display\n",
    "import re\n",
    "import os\n",
    "\n",
    "\n",
    "def generate_table_of_contents(filename):\n",
    "    # Open the notebook file\n",
    "    with open(filename, \"r\") as f:\n",
    "        notebook_content = f.read()\n",
    "\n",
    "    # Extract Markdown headers\n",
    "    headers = re.findall(r\"#+\\s+.*\", notebook_content)\n",
    "\n",
    "    # Generate Table of Contents\n",
    "    toc = []\n",
    "    for header in headers:\n",
    "        level = header.count(\"#\")  # Determine the header level\n",
    "        title = header.strip(\"#\").strip()  # Extract the title\n",
    "        link = (\n",
    "            re.sub(r\"[^\\w\\s]\", \"\", title).replace(\" \", \"-\").lower()\n",
    "        )  # Generate a link\n",
    "        toc.append(f\"{'  ' * (level - 1)}- [{title}](#{link})\")\n",
    "\n",
    "    # Display the Table of Contents\n",
    "    display(Markdown(\"\\n\".join(toc)))\n",
    "\n",
    "\n",
    "# Call the function to generate the Table of Contents\n",
    "example_folder = \"polars-exploration\"\n",
    "example_file = \"explore-polars.ipynb\"\n",
    "filename = os.path.join(os.getcwd(), example_folder, example_file)\n",
    "generate_table_of_contents(filename=filename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
