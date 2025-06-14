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

"""This script cleans the annotation data for the ElicitCam dataset."""

# Import required libraries.
import glob
import os

import pandas as pd


if __name__ == "__main__":

    # Create a list with terms to be removed from the annotation files.
    terms = [
        "No gesture performed.",
        "No hand gesture performed.",
        "The user does not perform any gesture.",
        "The user does not perform any hand gesture.",
        "The user does not perform any hand gestures.",
        "The user does not perform any gesture throughout the sequence.",
        "The user does not perform any distinct gesture.",
        "No discernible hand gesture is performed.",
    ]

    # Get the list of annotation files.
    filenames = glob.glob(
        os.path.join(
            os.path.dirname(__file__),
            "../data/descriptions/**/*.csv"
        ), recursive=True
    )
    filenames.sort()

    # Iterate over the annotation files.
    for filename in filenames:

        # Read the annotation file.
        df = pd.read_csv(filename)

        # Remove the terms from the annotation file.
        for term in terms:
            df = df.replace(term, pd.NA)

        # Remove the dot at the end of the descriptions.
        df = df.replace(r"\.$", "", regex=True)

        # Write the cleaned annotation file.
        df.to_csv(filename, index=False)
