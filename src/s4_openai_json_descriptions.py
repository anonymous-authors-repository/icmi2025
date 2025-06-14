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

"""This script generates a description of a hand gesture from a sequence of
JSON files with hand pose annotations using the OpenAI ChatGPT-4o."""

# Import required libraries.
import glob
import json
import os

import pandas as pd

from openai import AzureOpenAI, OpenAI


def get_openai_client() -> OpenAI:
    """Get the OpenAI client.

    Returns:
        OpenAI: The OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key for Microsoft Azure OpenAI not found in environment "
            "variables."
        )

    client = OpenAI(
        api_key=api_key
    )

    return client


def get_azure_openai_client() -> AzureOpenAI:
    """Get the OpenAI client via Microsoft Azure.

    Returns:
        AzureOpenAI: The Azure OpenAI client.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key for Microsoft Azure OpenAI not found in environment "
            "variables."
        )

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint is None:
        raise ValueError(
            "Azure endpoint for OpenAI not found in environment variables.")

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint
    )

    return client


def describe_hand_gesture(json_paths: list, stage: str) -> str:
    """
    Uploads a sequence of JSON to OpenAI ChatGPT-4o and generates a description.

    Args:
        json_paths (list): List of JSON file paths.
        stage (str): The stage of the hand gesture recognition process.

    Returns:
        str: The summarized description of the hand gesture.
    """

    # Get the OpenAI client.
    client = get_azure_openai_client() # or get_openai_client() for OpenAI API.

    # Read and combine the JSON files into a single string.
    combined_json_content = ""
    for i, json_path in enumerate(json_paths):

        # Read the JSON content from the file.
        with open(json_path, "r", encoding="utf-8") as file:
            json_content = json.load(file)

            # Filter the JSON content based on the stage.
            if stage == "combined":
                values = json.dumps(json_content)

            elif stage == "landmarks":
                values = json.dumps({
                    "hand_landmarks": json_content.get("hand_landmarks", {})
                })

            else:
                values = json.dumps({
                    k: v for k, v in json_content.items()
                        if k != "hand_landmarks"
                })

            combined_json_content += f"Image {i + 1}: {values}\n"

    # Create the GPT-4 API prompt.
    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant trained to analyze a sequence of JSON files (i.e., hand pose annotations extracted from a sequence of images using the Google MediaPipe Hands model) from a video clip and describe the user's hand gestures. Describe concisely the identified hand gestures in a one-sentence. Don't describe the meaning of the gesture, indication, or suggestion of use, nor the user's intention. If the user does not perform any gesture, report this. Consider the development of the gesture over the sequence of JSON files, not each JSON file individually.
            """
        },
        {
            "role": "user",
            "content":  """
Here are hand annotations from a video clip where the user is performing none, one single or two combined hand gestures in sequence. Ignore the hands' initial and final resting positions. Focus on describing the fingers and hand pose, orientation and direction, and the main movements that characterize the action gestures.
            """
        },
        {
            "role": "user",
            "content": combined_json_content
        }
    ]

    # Call the GPT-4 model.
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=200,
        temperature=0,
        top_p=0.1
    )

    # Extract the response content.
    return response.choices[0].message.content


if __name__ == "__main__":

    # Define the stages to process.
    for i, stage in enumerate(["poses", "landmarks", "combined"]):

        # Define the file path and column names.
        file_path = os.path.join(
            os.path.dirname(__file__),
            f"../data/descriptions/d{i + 4}_openai_{stage}_descriptions.csv"
        )

        columns = [ f"c{j}_description" for j in range(1, 9) ]
        columns.insert(0, "id_video")  # Add "id_video" as the first column.

        # Load the existing annotations or create a new DataFrame.
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=columns).astype(str)

        # Create a list of all the folders in the JSON directory.
        json_folder = os.path.join(
            os.path.dirname(__file__),
            "dataset/jsons"
        )

        folders = [ f.path for f in os.scandir(json_folder) if f.is_dir() ]
        folders = []
        for root, dirs, _ in os.walk(json_folder):
            for d in dirs:
                folders.append(os.path.join(root, d))
        folders = sorted(folders)

        # Process each folder in the list.
        for folder in folders:

            # Get the list of JSON files in the current folder.
            filenames = sorted(glob.glob(os.path.join(folder, "*.json")))

            # Skip the folder if there are no JSON files.
            if not filenames:
                continue

            # Extract the "id_video" and "description ID" from the folder path.
            id_video = folder.split("/")[-2]
            id_description = folder.split("/")[-1]

            # Ensure the id_video exists in the DataFrame.
            if id_video not in df["id_video"].values:
                df = pd.concat([
                    df, pd.DataFrame({ "id_video": [id_video] })
                ], ignore_index=True)

            # Check if the folder has not already been annotated.
            DESCRIPTION_COL = f"{id_description}_description"
            print(f"Processing {id_video} / description {id_description}...")

            if df.loc[df["id_video"] == id_video, DESCRIPTION_COL].isna().all():

                # Get the sample files from the list of JSON files.
                try:
                    result = describe_hand_gesture(filenames, stage)

                # Handle the case where the content is blocked.
                except RuntimeError as e:
                    print(f"Exception Error: {str(e)}.")
                    continue

                # Add the result to the corresponding row and column in the
                # DataFrame.
                df.loc[df["id_video"] == id_video, DESCRIPTION_COL] = result

                # Save the updated DataFrame to the CSV file.
                df.to_csv(file_path, index=False)

        # Order the DataFrame by "id_video".
        df = df.sort_values(by="id_video").reset_index(drop=True)

        # Save the ordered DataFrame to the CSV file.
        df.to_csv(file_path, index=False)
