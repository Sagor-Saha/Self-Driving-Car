
# Self Driving Car

An autonomous car (also known as a driverless car, self-driving car, and robotic car) is a vehicle that is capable of sensing its environment and navigating without human input. Autonomous cars combine a variety of techniques to perceive their surroundings, including radar, laser light, GPS, odometry, and computer vision. Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage

## Objective 
Given images of road you need to predict its degree of turning.
 
## Inspiration 🗼

1) [Udacity Self driving car](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
2) [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

## Code Requirements 🦄
- `pip install requirements.txt`

### Dataset 1
Approximately 45,500 images, 2.2GB. One of the original datasets I made in 2017. Data was recorded around Rancho Palos Verdes and San Pedro California.

[Link](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing)

Data format is as follows: `filename.jpg angle`

### Dataset 2

Approximately 63,000 images, 3.1GB. Data was recorded around Rancho Palos Verdes and San Pedro California.

[Link](https://drive.google.com/open?id=1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B)

Data format is as follows: `filename.jpg angle,year-mm-dd hr:min:sec:millisec`

`if you use second dataset, you need to convert it in the form of` `filename.jpg angle`


Use `python train.py` to train the model

Use `python run.py` to run the model on a live webcam feed

Use `python run_dataset_C.py` to run the model on the dataset

You will see

![video](https://drive.google.com/uc?export=view&id=1Y0G9XoT7FDtAHjJoRTPB4D2iKnAudMLu)

Use `python app.py` if you want to see running it on flask
then enter into the url `http://127.0.0.1:5000/`.After that you see

![image](https://drive.google.com/uc?export=view&id=1ic2dSztmAkp7DmM4IfLwAmhHW2sSiBz-)

when you click show demo

![image](https://drive.google.com/uc?export=view&id=1VtuNft5QudP5bjNEmqTxVbNT5GtWgvVg)

## File Organization 🗄️
```shell
├── selfDrivingCar (Current Directory)
        ├── deploy
        ├── static
        ├── templates
        ├── app.py
        ├── driving_data.py
        ├── modelckpt
        ├── model.py
        ├── requirements.txt
        ├── run_dataset_C.py
        ├── steering_wheel_image.jpg
        ├── train.py
        └── Readme.Md
```

## References 🔱
 
 - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
 - [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 
 - This implementation also took a lot of inspiration from the Sully Chen github repository: https://github.com/SullyChen/Autopilot-TensorFlow  

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sagorsaha.tech/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sagor-saha-047001111/)
