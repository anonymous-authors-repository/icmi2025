#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2025 Anonymous Authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
This script processes human-annotated structured descriptions from a CSV file.
It loads the raw annotation data, filters for specific description columns,
replaces any "unknown" values with NaN, sorts the entries by video ID, and
saves the cleaned and ordered data to a new CSV file for further analysis.
"""

# Import required libraries.
import os

import pandas as pd


if __name__ == "__main__":

    # Define the file path and column names.
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../dataset/elicit_cam.csv"
    )
    columns = [ f"c{i}_description" for i in range(1, 9) ]
    columns.insert(0, "id_video")  # Add "id_video" as the first column.

    # Load the existing annotations.
    df = pd.read_csv(file_path)

    # Filter the columns to only include the "description" columns.
    df = df[columns]

    # Replace "unknown" values with NaN.
    df = df.replace(r"\s*unknown\s*", pd.NA, regex=True)

    # Order the DataFrame by "id_video".
    df = df.sort_values(by="id_video").reset_index(drop=True)

    # Save the ordered DataFrame to the CSV file.
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../data/descriptions/d0_human_structured_descriptions.csv"
    )

    df.to_csv(file_path, index=False)
