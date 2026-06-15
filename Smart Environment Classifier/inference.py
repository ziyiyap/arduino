import serial
import torch
from train import NeuralNetwork, idx_class_y1, idx_class_y2, idx_class_y3

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

#Serial
ser = serial.Serial(port="COM3", baudrate=9600)

checkpoint = torch.load("models.pth")

NN_presence = NeuralNetwork("presence", 1, 8, 2).to("cuda")
NN_presence.load_state_dict(checkpoint["presence"])
NN_env = NeuralNetwork("env", 3, 16, 4).to("cuda")
NN_env.load_state_dict(checkpoint["env"])
NN_sound = NeuralNetwork("sound", 1, 8, 2).to("cuda")
NN_sound.load_state_dict(checkpoint["sound"])

mean_X1, std_X1 = checkpoint["mean_presence"], checkpoint["std_presence"]
mean_X2, std_X2 = checkpoint["mean_env"], checkpoint["std_env"]
mean_X3, std_X3 = checkpoint["mean_sound"], checkpoint["std_sound"]
### Output serial
while True:
    raw = ser.readline()
    data_row = [float(data.strip()) for data in raw.decode("utf-8").split(",")]

    distance = data_row[-1]
    env = [data_row[i] for i in range(3)]
    sound = data_row[-2]



    y_pred_presence = NN_presence.predict(NN_presence.standardize(torch.tensor(distance).float(), mean_X1, std_X1), idx_class_y1)
    y_pred_env = NN_env.predict(NN_env.standardize(torch.tensor(env).float(), mean_X2, std_X2), idx_class_y2)
    y_pred_sound = NN_sound.predict(NN_sound.standardize(torch.tensor(sound).float(), mean_X3, std_X3), idx_class_y3)

    msg = f"{y_pred_presence}, {y_pred_env}, {y_pred_sound}\n"
    ser.write(msg.encode())


    print(data_row)