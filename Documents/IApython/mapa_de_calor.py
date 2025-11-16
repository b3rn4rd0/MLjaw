import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
import timm
import os


MODEL_PATH = r"modelo_inception_dtm_final.pth"  # rota do modelo salvo
IMAGE_PATH = r"" # a rota para testar a radiografia
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class InceptionV4_DTM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(InceptionV4_DTM, self).__init__()
        self.base_model = timm.create_model('inception_v4', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        out = self.classifier(features)
        return out



model = InceptionV4_DTM(num_classes=2).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)


cam_extractor = SmoothGradCAMpp(model)
out = model(input_tensor)
pred_class = out.argmax(dim=1).item()
activation_map = cam_extractor(pred_class, out)


plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.imshow(activation_map[0].squeeze().cpu().numpy(), cmap='jet', alpha=0.5)
plt.title(f"Classe predita: {pred_class} (0=Normal, 1=DTM)")
plt.axis("off")

os.makedirs("resultados_heatmap", exist_ok=True)
output_path = os.path.join("resultados_heatmap", "heatmap_exemplo.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Mapa de calor salvo em: {output_path}")
