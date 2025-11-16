import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Mesma arquitetura do modelo
class InceptionV4_DTM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(InceptionV4_DTM, self).__init__()
        self.base_model = timm.create_model('inception_v4', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        out = self.classifier(features)
        return out

def carregar_modelo(filepath="modelo_inception_dtm_final.pth"):
    """
    Carrega o modelo salvo
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Criar modelo com mesma arquitetura
    model = InceptionV4_DTM(
        num_classes=checkpoint['num_classes'], 
        pretrained=False
    ).to(device)
    
    # Carregar pesos treinados
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo carregado de: {filepath}")
    print(f"Treinado por {checkpoint['epoch']} épocas")
    
    return model, checkpoint

def inferencia_imagem(model, imagem_path, patch_size=299):
    """
    Faz inferência em uma única imagem
    """
    # Transformações (mesmas do treinamento)
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar e processar imagem
    imagem = Image.open(imagem_path).convert('RGB')
    imagem_tensor = transform(imagem).unsqueeze(0).to(device)
    
    # Inferência
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            output = model(imagem_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
    
    # Mapear classe para nome
    class_names = {0: "Normal", 1: "DTM"}
    
    return {
        'classe': class_names[predicted_class],
        'classe_id': predicted_class,
        'confianca': confidence * 100,
        'probabilidades': {
            'Normal': probabilities[0][0].item() * 100,
            'DTM': probabilities[0][1].item() * 100
        }
    }

def inferencia_pasta(model, pasta_path, patch_size=299):
    """
    Faz inferência em todas as imagens de uma pasta
    """
    resultados = []
    
    # Extensões de imagem suportadas
    extensoes = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for arquivo in os.listdir(pasta_path):
        if arquivo.lower().endswith(extensoes):
            caminho_completo = os.path.join(pasta_path, arquivo)
            print(f"\nProcessando: {arquivo}")
            
            resultado = inferencia_imagem(model, caminho_completo, patch_size)
            resultado['arquivo'] = arquivo
            resultados.append(resultado)
            
            print(f"  Predição: {resultado['classe']}")
            print(f"  Confiança: {resultado['confianca']:.2f}%")
            print(f"  Prob. Normal: {resultado['probabilidades']['Normal']:.2f}%")
            print(f"  Prob. DTM: {resultado['probabilidades']['DTM']:.2f}%")
    
    return resultados

# Exemplo de uso
if __name__ == "__main__":
    # Carregar modelo
    modelo_path = "modelo_inception_dtm_final.pth"
    model, checkpoint = carregar_modelo(modelo_path)
    
    
    imagem_teste = r"" #rota para usar a radiografia testada
    
    if os.path.exists(imagem_teste):
        resultado = inferencia_imagem(model, imagem_teste, checkpoint['patch_size'])
        print("\n=== Resultado da Inferência ===")
        print(f"Classe predita: {resultado['classe']}")
        print(f"Confiança: {resultado['confianca']:.2f}%")
        print(f"Probabilidades:")
        print(f"  Normal: {resultado['probabilidades']['Normal']:.2f}%")
        print(f"  DTM: {resultado['probabilidades']['DTM']:.2f}%")
    