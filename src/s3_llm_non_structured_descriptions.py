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
images using the OpenAI ChatGPT-4o."""

# Import required libraries.
import base64
import glob
import os

import cv2 as cv
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


def describe_hand_gesture(image_paths: list) -> str:
    """
    Uploads a sequence of images to OpenAI ChatGPT-4o and generates a
    description of the hand gesture.

    Args:
        image_paths (list): List of images file paths.

    Returns:
        str: The summarized description of the hand gesture.
    """

    # Get the OpenAI client.
    client = get_azure_openai_client() # or get_openai_client() for OpenAI API.

    # Prepare the image input for GPT-4.
    image_inputs = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{ encode_image(img_path) }"
            }
        } for img_path in image_paths
    ]

    # Create the GPT-4 API prompt
    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant trained to analyze a sequence of images from a video clip and describe the user's hand gestures. Describe concisely the identified hand gestures in a one-sentence. Don't describe the meaning of the gesture, indication, or suggestion of use, nor the user's intention. If the user does not perform any gesture, report this. Consider the development of the gesture over the sequence of images, not each image individually.
            """
        },
        {
            "role": "user",
            "content": """
Here are the images in which the user performs none, one single, or two combined hand gestures in sequence. Ignore the hands' initial and final resting positions. Focus on describing the fingers and hand pose, orientation and direction, and the main movements that characterize the action gestures. Include hand interactions with the head parts if they occur.
            """
        },
        {
            "role": "user",
            "content": image_inputs
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


def encode_image(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        str: Base64 encoded string of the image.
    """

    # Read and resize the image file.
    image = cv.imread(image_path)
    resized_image = cv.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Encode the image as JPEG.
    success, buffer = cv.imencode(".jpg", resized_image)
    if not success:
        raise ValueError("Failed to encode image to JPEG format.")

    # Convert the encoded image to Base64.
    return base64.b64encode(buffer).decode("utf-8")


if __name__ == "__main__":

    # Define the file path and column names.
    file_path = os.path.join(
        os.path.dirname(__file__),
        "../data/descriptions/d3_llm_non_structured_descriptions.csv"
    )

    columns = [ f"c{i}_description" for i in range(1, 9) ]
    columns.insert(0, "id_video")  # Add "id_video" as the first column.

    # Load the existing annotations or create a new DataFrame.
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns).astype(str)

    # Create a list of all the folders in the images directory.
    image_folder = os.path.join(
        os.path.dirname(__file__),
        "dataset/images/"
    )

    folders = [ f.path for f in os.scandir(image_folder) if f.is_dir() ]
    folders = []
    for root, dirs, _ in os.walk(image_folder):
        for d in dirs:
            folders.append(os.path.join(root, d))
    folders = sorted(folders)

    # Process each folder in the list.
    for folder in folders:

        # Get the list of image files in the current folder.
        filenames = sorted(glob.glob(os.path.join(folder, "*.[pj][np]g")))

        # Skip the folder if there are no image files.
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

        if (df.loc[df["id_video"] == id_video, DESCRIPTION_COL].isna().all()):

            # Get the sample files from the list of image files.
            try:

                # Limit the number of files to 50 for faster processing.
                if len(filenames) > 50:
                    SAMPLE_SIZE = 50
                    file_count = len(filenames)

                    indices = []
                    for i in range(SAMPLE_SIZE):
                        indices.append(
                            int((file_count - 1) * i / (SAMPLE_SIZE - 1)))

                    filenames = [ filenames[i] for i in indices ]
                    filenames.sort()

                # Generate the description of the hand gesture.
                result = describe_hand_gesture(filenames)

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
