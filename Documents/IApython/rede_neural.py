#uso da rede neural inception v4
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" Usando dispositivo: {device}")

PATCH_SIZE = 299   # Inception-v4 requer 299x299
BATCH_SIZE = 8
NUM_CLASSES = 2    # 0 = normal, 1 = DTM
LR = 1e-4
EPOCHS = 29

# rotas para os datasets
DATASET_ROOT = r"C:\Users\pichombas\Documents\pasta de raio x"
TRAIN_DIR = os.path.join(DATASET_ROOT, "treino_patches_frontal")
VAL_DIR = os.path.join(DATASET_ROOT, "validação_patches_frontal")
TEST_DIR = os.path.join(DATASET_ROOT, "teste_patches_frontal")

#processamento e transformações

transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#datasets e dataloaders

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory= True, persistent_workers= True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory= True, persistent_workers= True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory= True, persistent_workers= True)

print(f"Total de patches - Treino: {len(train_dataset)}, Validação: {len(val_dataset)}, Teste: {len(test_dataset)}")

# função da arquitetura Inception v4

class InceptionV4_DTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(InceptionV4_DTM, self).__init__()
        self.base_model = timm.create_model('inception_v4', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        out = self.classifier(features)
        return out

# função para o treinamento

def treinamento(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        print("teste")
        start_time = time.time()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(train_acc)

        # Avaliação no conjunto de validação
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Época [{epoch+1}/{epochs}] | Tempo: {epoch_time:.2f}s | "
              f"Loss Treino: {train_losses[-1]:.4f} | Acc Treino: {train_acc:.2f}% | "
              f"Loss Val: {val_losses[-1]:.4f} | Acc Val: {val_acc:.2f}%")

        model.train()
        torch.cuda.empty_cache()

    return train_losses, val_losses, train_accs, val_accs

# função de teste e validação

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusão:")
    print(cm)
    print(f"\n Acurácia final no conjunto de teste: {acc*100:.2f}%")
    return cm, acc


# Função para salvar o modelo
def salvar_modelo(model, optimizer, epoch, train_losses, val_losses, filepath="modelo_inception_dtm.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_classes': NUM_CLASSES,
        'patch_size': PATCH_SIZE
    }, filepath)
    print(f"\nModelo salvo em: {filepath}")


# execuçã da função principal

if __name__ == "__main__":
    model = InceptionV4_DTM(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, train_accs, val_accs = treinamento(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS
    )

    # Salvar o modelo na pasta Modelos/recente
    salvar_modelo(
        model, optimizer, EPOCHS, train_losses, val_losses, 
        filepath=r"C:\Users\pichombas\Documents\Modelos\recente\modelo_inception_dtm_final.pth"
    )

    # Avaliação final
    cm, acc = evaluate_model(model, test_loader)

    # Gráficos de desempenho
    plt.figure()
    plt.plot(train_losses, label="Treino")
    plt.plot(val_losses, label="Validação")
    plt.title("Evolução da Função de Perda")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="Treino")
    plt.plot(val_accs, label="Validação")
    plt.title("Evolução da Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia (%)")
    plt.legend()
    plt.show()
