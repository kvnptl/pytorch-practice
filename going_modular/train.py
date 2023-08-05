"""
Trains a PyTorch image classification model using device agnostic code.
"""

import os
import torch
from torchvision import datasets, transforms
import data_setup, engine, model_builder, utils

NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create simple transform
data_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create a dataloader and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                                test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(input_shape=3, 
                                hidden_units=HIDDEN_UNITS, 
                                output_shape=len(class_names)).to(device)

# Setup loss function, optimizer and scheduler
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), 
                            lr=LEARNING_RATE)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Start training
engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS, device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save model
utils.save_model(model=model, target_dir="models", model_name="05_going_modular_script_mode_tingvgg_model.pth")
