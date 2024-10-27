import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model and processor
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model.to(device)
# preprocessor.to(device)

def calculate_similarity(image, name, text_prompt):
    image_path = f'./images/{name}.png' 
    image.save(image_path)

    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move processed tensors to the device

    # Forward pass
    outputs = model(**inputs)

    # Normalize and calculate cosine similarity
    image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

    result_text = f"According to OpenCLIP, the image and the text prompt have a cosine similarity of {cosine_similarity.item() * 100:.2f}%."
    sim = round(cosine_similarity.item() * 100, 2)
    result = {
        "name": name,
        "image": image_path,
        "text": text_prompt,
        "similarity": sim
    }
    df = pd.read_csv('./submissions.csv')
    df.reset_index(drop=True, inplace=True)  # Drop any existing index column
    df = df._append(result, ignore_index=True)
    df.to_csv('./submissions.csv', index=False)

    result_df = df[['name','similarity']].sort_values(by=['similarity'], ascending=False)
    result_df.reset_index(drop=True, inplace=True)
    result_df.rename(columns={'index': 'rank'}, inplace=True)
    result_df['rank'] = result_df.index + 1

    print(result_df.head())

    return result_text, result_df


with gr.Blocks() as demo:
# Set up Gradio interface
    iface = gr.Interface(
        fn=calculate_similarity,
        inputs=[
            gr.Image(type="pil", label="Upload Image", height=512),
            gr.Textbox(label="Name"),
            gr.Textbox(label="Text Prompt"),
        ],
        outputs=[gr.Textbox(label="Result", show_label=True), gr.Dataframe(label='Leaderboard', type='pandas', show_label=True, headers=['rank','name','similarity'])],
        allow_flagging="never",
        title="OpenClip Similarity Calculator",
        description="Upload an image and provide a text prompt to calculate the similarity."
    )


if __name__ == "__main__":
    demo.launch()

