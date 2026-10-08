import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import warnings

warnings.filterwarnings("ignore") # Suppress scikit-learn warnings for cleaner output

def prepare_classical_data(csv_path, num_tabular_features=15):
    """
    PHASE 1 & 2: Data Alignment and Classical Feature Selection.
    """
    print("--- Starting Phase 1 & 2: Data Prep & Classical Feature Selection ---")
    
    df = pd.read_csv(csv_path)
    print(f"Initial rows in CSV: {len(df)}")
    
    # --- NEW FIX: CHECK IF IMAGES ACTUALLY EXIST ON YOUR DRIVE ---
    print("Scanning hard drive for physical image files... (This might take a few seconds)")
    df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
    
    # Keep only the rows where the image actually exists!
    df = df[df['image_exists'] == True].drop(columns=['image_exists'])
    
    print(f"Valid rows remaining (Images Found): {len(df)}")
    if len(df) == 0:
        raise ValueError("CRITICAL ERROR: No images were found on your hard drive! Check your BASE_IMAGE_DIR in the aligner script.")
    # -------------------------------------------------------------

    # Separate features and labels (Ignore text and paths)
    feature_cols = [col for col in df.columns if col not in ['image_path', 'label', 'text_augment']]
    X_tabular = df[feature_cols]
    y = df['label']

    print(f"Original Tabular Shape: {X_tabular.shape} (Lots of noise!)")
    
    # 1. Clean: Handle missing values and scale
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    X_cleaned = scaler.fit_transform(imputer.fit_transform(X_tabular))
    
    # 2. Select: Classical Feature Selection
    selector = SelectKBest(score_func=f_classif, k=num_tabular_features)
    X_selected = selector.fit_transform(X_cleaned, y)
    
    print(f"Reduced Tabular Shape: {X_selected.shape} (Top {num_tabular_features} features selected classically)")
    
    # Rebuild the master dataframe
    master_df = pd.DataFrame(X_selected, columns=[f'selected_feat_{i}' for i in range(num_tabular_features)])
    master_df['image_path'] = df['image_path'].values # Reset index to match
    master_df['label'] = y.values
    
    # Save the indices
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]
    
    return master_df, num_tabular_features, selected_feature_names

class MultimodalPneumoniaDataset(Dataset):
    """
    PHASE 3: The Data Loader. 
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        self.image_paths = self.dataframe['image_path'].values
        self.labels = self.dataframe['label'].values
        self.tabular_features = self.dataframe.drop(['image_path', 'label'], axis=1).values.astype(np.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # We shouldn't hit this anymore because of the Pandas filter, but kept as a safety net!
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        tab_data = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, tab_data, label

class MultimodalFusionNet(nn.Module):
    """
    PHASE 4: The Fusion Architecture.
    """
    def __init__(self, num_tabular_features, num_classes=2):
        super(MultimodalFusionNet, self).__init__()
        
        # --- VISION BRANCH ---
        resnet = models.resnet18(pretrained=True)
        self.vision_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_out_features = 512 
        
        # --- TABULAR BRANCH ---
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), 
            nn.ReLU()
        )
        self.tabular_out_features = 32
        
        # --- FUSION LAYER ---
        combined_features = self.vision_out_features + self.tabular_out_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) 
        )

    def forward(self, image, tabular):
        vision_out = self.vision_branch(image)
        vision_out = torch.flatten(vision_out, 1) 
        tabular_out = self.tabular_branch(tabular)
        fused = torch.cat((vision_out, tabular_out), dim=1)
        output = self.classifier(fused)
        return output

def train_model(model, train_loader, criterion, optimizer, num_epochs=8, device='cpu'):
    """
    PHASE 5: Training
    """
    print(f"\n--- Starting Phase 5: Training on {device.upper()} ---")
    model.to(device)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_idx, (images, tab_data, labels) in enumerate(train_loader):
            images, tab_data, labels = images.to(device), tab_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, tab_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total_preds
        epoch_acc = (correct_preds / total_preds) * 100.0
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    print("Training Complete. Model saved as 'baseline_multimodal_model.pth'")
    torch.save(model.state_dict(), 'baseline_multimodal_model.pth')
    return history

def evaluate_and_log(model, test_loader, device, feature_names, history):
    """
    PHASE 6: Evaluation and Logging (CSV Output)
    """
    print("\n--- Starting Phase 6: Evaluation & Logging ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, tab_data, labels in test_loader:
            images, tab_data, labels = images.to(device), tab_data.to(device), labels.to(device)
            outputs = model(images, tab_data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    
    # 1. Save Classification Report as CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("baseline_classification_report.csv")
    
    # 2. Save Training History as CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv("baseline_training_history.csv", index_label="epoch")
    
    # 3. Save Selected Features as CSV
    features_df = pd.DataFrame({"classically_selected_features": feature_names})
    features_df.to_csv("baseline_selected_features.csv", index=False)
        
    print("Results successfully saved to CSV files: 'baseline_classification_report.csv', 'baseline_training_history.csv', 'baseline_selected_features.csv'")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 1. Prepare Data (Reads your aligned dataset and FILTERS missing images!)
    MASTER_CSV_PATH = "master_dataset.csv" 
    master_df, num_features, selected_features = prepare_classical_data(csv_path=MASTER_CSV_PATH)
    
    # 2. Split into Train/Test
    train_df, test_df = train_test_split(master_df, test_size=0.2, random_state=42)
    
    # 3. Create Datasets and Loaders
    train_dataset = MultimodalPneumoniaDataset(train_df, transform=transform)
    test_dataset = MultimodalPneumoniaDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 4. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalFusionNet(num_tabular_features=num_features, num_classes=2)
    
    # 5. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Train!
    history = train_model(model, train_loader, criterion, optimizer, num_epochs=8, device=device)
    
    # 7. Evaluate and Log Results
    evaluate_and_log(model, test_loader, device, selected_features, history)