import clip
from PIL import Image

def image_caption(image_path):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Encode the image
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Prepare the input text
    input_text = ["a photo of a", "a picture of a", "an image of a"]

    # Generate captions
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize(input_text).to(device))

        # Calculate the similarity between text and image features
        logits_per_image, logits_per_text = model(image, text_features)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the label names from the CLIP tokenizer
    label_names = clip.tokenize(input_text).split(" ")

    # Get the highest probability caption
    caption = input_text[probs.argmax()]

    return caption

if __name__ == "__main__":
    image_path = r"C:\Users\HP\Downloads\guip165 (1)\static\outputs\test.jpg"  # Provide the path to your image
    caption = image_caption(image_path)
    print("Caption:", caption)
