import torch
from dataLoaders import InferLoader
import utils.helpermethods as helpermethods
from Models import LeNet_5
import multiprocessing
from torchvision import transforms


def main(queue: multiprocessing.Queue = multiprocessing.Queue()) -> None:  # type: ignore
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    print('Infering...')
    model = LeNet_5.Model(10)
    checkpoint_path = 'E:/ML/Checkpoints/a.pt'
    infer_path = './src/Website/Temp'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    infer_data = InferLoader.CustomDataset(infer_path, transform)
    infer_loader: torch.utils.data.DataLoader[InferLoader.CustomDataset] = torch.utils.data.DataLoader(infer_data, batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for inputs in infer_loader:
            # Move the data to the GPU if available
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
    predictions = [labels[prediction] for prediction in predictions]
    helpermethods.send_result(queue, predictions)


if __name__ == '__main__':
    main()
