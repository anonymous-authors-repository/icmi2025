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

"""This script analyzes the hand gestures performed in the unified
communication platform context during hybrid meetings. It uses the OpenAI
ChatGPT-4o model to generate a description of the hand gestures performed by
the participants and try to identify the corresponding hand gesture command."""

# Import required libraries.
import glob
import os
import re

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


def predict_ucp_command(description: str) -> str:
    """
    Uploads a hand gesture description to OpenAI ChatGPT-4o and predicts an UCP
    command.

    Args:
        description (str): The description of the hand gesture.

    Returns:
        str: The unified communication platform command predicted by the model.
    """

    # Get the OpenAI client.
    client = get_azure_openai_client() # or get_openai_client() for OpenAI API.

    # Define the user message based on the stage.
    user_message = """
Here is a description of a user's hand gesture. A system can recognize the user's gesture as a controlling command in the given system or application context. This gesture refers to commands performed by a user participating in a hybrid meeting. Consider that the user is physically in a room and remotely connected to other users through a unified communication platform. Evaluate the gesture according to the following options of control commands: - `Increase volume` - `Decrease volume` - `Mute microphone` - `Unmute microphone` - `Turn off camera` - `Turn on camera` - `Ask for a question` - `End call`.
    """

    # Create the GPT-4 API prompt
    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant trained to interpret a human hand gesture by analyzing its textual description. Consider that a system can recognize the user's gesture as a command. You should determine the user's intention to use the gesture as a controlling command for a given context or scenario. Present a precise label for the identified command (user intention). Avoid unnecessary words or special characters, and maintain consistency."
            """
        },
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "user",
            "content": description
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
    response_content = response.choices[0].message.content
    return re.sub(r'[^a-zA-Z]', ' ', response_content).strip().capitalize()


if __name__ == "__main__":

    # Get the list of annotation files.
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../dataset/descriptions"
    )
    filenames = glob.glob(file_path + "**/*.csv", recursive=True)
    filenames.sort()

    # Define the hand gesture descriptions.
    columns = [ f"c{i}_command" for i in range(1, 9) ]
    columns.insert(0, "id_video")  # Add "id_video" as the first column.

    # Process each annotation file.
    for filename in filenames:

        # Load the hand gesture descriptions.
        df_annotations = pd.read_csv(filename)

        # Load the existing annotations or create a new DataFrame.
        file_path = filename.replace(
            "/descriptions/", "/predictions/"
        ).replace(
            "_descriptions.", "_predictions."
        )

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

        else:
            destination_folder = os.path.dirname(file_path)
            os.makedirs(destination_folder, exist_ok=True)
            df = pd.DataFrame(columns=columns).astype(str)

        # Process each row in the annotation file.
        for index, row in df_annotations.iterrows():

            # Ensure the id_video exists in the DataFrame.
            id_video = row["id_video"]
            if id_video not in df["id_video"].values:
                df = pd.concat([
                    df, pd.DataFrame({ "id_video": [id_video] })
                ], ignore_index=True)

            # Analyze the hand gesture descriptions.
            for id_description, description in enumerate(row.values[1:]):

                # Skip the row if the description is empty.
                if pd.isna(description):
                    continue

                # Get the column names for the description and command.
                COMMAND_COL = f"c{id_description + 1}_command"
                print(
                    f"Processing {id_video} / description {COMMAND_COL}...")

                # Check if the folder has not already been annotated.
                if df.loc[
                    df["id_video"] == id_video, COMMAND_COL].isna().all():

                    # Get the descriptions from the GPT-4 model.
                    try:
                        result = predict_ucp_command(description)
                        if result is None:
                            continue

                    # Handle the case where the content is blocked.
                    except RuntimeError as e:
                        print(f"Exception Error: {str(e)}.")
                        continue

                    # Add the result to the corresponding row and column in
                    # the DataFrame.s
                    df.loc[df["id_video"] == id_video, COMMAND_COL] = \
                        result.strip()

                    # Save the updated DataFrame to the CSV file.
                    df.to_csv(file_path, index=False)

        # Order the DataFrame by "id_video".
        df = df.sort_values(by="id_video").reset_index(drop=True)

        # Save the ordered DataFrame to the CSV file.
        df.to_csv(file_path, index=False)
