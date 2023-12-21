import argparse
import os
import torch
from torch import optim
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from resnet_model import resnet101
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix

def collate_fn(batch):
    return batch

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root

        self.image_path = [os.path.join(root, 'result', img) for img in
                           os.listdir(os.path.join(root, 'result')) if img.endswith(('.jpg', '.jpeg', '.png'))]

        self.label_dir = [os.path.join(root, 'label', label) for label in
                          os.listdir(os.path.join(root, 'label')) if label.endswith('.txt')]

        self.labels = [self.read_label(label_path) for label_path in self.label_dir]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img_path = self.image_path[index]
        label_path = self.label_dir[index]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            img = torch.zeros((3, 224, 224), dtype=torch.float32)  # Create a default image
            label = torch.tensor([0])  # Use zero tensor as the default label
            return img, label

        label = self.labels[index]

        if label is None:
            label = torch.tensor([0])  # Use zero tensor as the default label

        return img, label

    def read_label(self, label_path):
        try:
            with open(label_path, 'r') as file:
                lines = file.readlines()

            if lines:
                label = int(lines[0].strip())
                return label

            return None
        except Exception as e:
            print(f"Error reading label {label_path}: {str(e)}")
            return None

def val(model, criterion, val_dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())

        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Calculate and print accuracy
        acc = accuracy_score(all_labels, all_predictions)
        print("Accuracy: {:.4f}".format(acc))

        # Calculate and print recall
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)  # or use average='micro'
        print("Recall: {:.4f}".format(recall))

        # Calculate and print average loss
        avg_loss = total_loss / len(val_dataloader)
        print("Average Loss: {:.4f}".format(avg_loss))

def train(args, train_dataloader, val_dataloader):
    resnet = resnet101(num_classes=args.num_classes)
    if args.pretrained_model:
        resnet.load_state_dict(torch.load(args.pretrained_model))
    resnet.to("cuda")
    print(resnet)

    epochs = args.epoch
    learning_rate = args.learning_rate
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to("cuda")
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

    total_train_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for data in train_dataloader:
            total_train_step = total_train_step + 1
            resnet.train()
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())

            if total_train_step % 10 == 0:
                print("Training step {} , Loss: {}".format(total_train_step, running_loss))
                running_loss = 0.0
                conf_matrix = confusion_matrix(all_labels, all_predictions)
                acc = accuracy_score(all_labels, all_predictions)
                # Calculate and print recall
                recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)  # or use average='micro'
                print("Epoch {}: Accuracy: {:.4f}, Recall: {:.4f}".format(epoch + 1, acc, recall))

            torch.save(resnet.state_dict(), args.save_path)
            if total_train_step % 100 == 0:
                val(args, resnet, criterion, val_dataloader)  # Evaluate the model on the validation set after each training epoch

def test(args, test_dataloader):
    resnet = resnet101(num_classes=args.num_classes)
    if args.pretrained_model:
        resnet.load_state_dict(torch.load(args.pretrained_model))
    resnet.to("cuda")
    print(resnet)

    epochs = args.epoch
    learning_rate = args.learning_rate
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to("cuda")
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

    test_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for data in test_dataloader:
            test_step += 1
            resnet.train()
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, 1).cpu().numpy())

            if test_step % 10 == 0:
                print("Test step {} , Loss: {}".format(test_step, running_loss))
                running_loss = 0.0
                conf_matrix = confusion_matrix(all_labels, all_predictions)
                acc = accuracy_score(all_labels, all_predictions)
                # Calculate and print recall
                recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)  # or use average='micro'
                print("Epoch {}: Accuracy: {:.4f}, Recall: {:.4f}".format(epoch + 1, acc, recall))

        torch.save(resnet.state_dict(), args.save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Parameters to train net')
    parser.add_argument('--epoch', default=10, type=int, help='Epochs to train the network')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Base value of learning rate.')
    parser.add_argument('--train_batch_size', default=8, type=int, help='Training batch size.')
    parser.add_argument('--pretrained_model', default='', type=str, help='Pretrained base model')
    parser.add_argument('--num_classes', default=32, type=int, help='Number of classes')
    parser.add_argument("--save_path", default="./models/resnet101.pth", help="Model save path")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print('CUDA available:', torch.cuda.is_available())
    args = get_args()

    # Use the same dataset transformation
    dataset_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = CustomDataset(root='D:\\Desktop\\learn_tongue\\舌象集合\\舌象\\database_tongue_add',
                                 transform=dataset_transform)

    # Randomly split the dataset into training (80%), validation (10%), and test (10%) sets
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=42)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
    )

    # Model creation and training
    resnet = resnet101(num_classes=args.num_classes)
    if args.pretrained_model:
        resnet.load_state_dict(torch.load(args.pretrained_model))
    resnet.to("cuda")

    # Train the model with the modified data split
    train(args, train_dataloader, val_dataloader)
    test(args, test_dataloader)
