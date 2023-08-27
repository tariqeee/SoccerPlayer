# SoccerPlayer YOLOv8 Detection Model

Welcome to the SoccerPlayer Detection project! At the core of this project is the advanced YOLOv8 (You Only Look Once) object detection model, which has been tailored to identify soccer players in various media formats, including images and videos. YOLOv8 stands out for its speed and accuracy, making it an ideal choice for real-time object detection tasks.



Whether you are an enthusiast looking to experiment with object detection, or a developer aiming to integrate this into a broader application, this README is your comprehensive guide. Herein, we provide you with detailed steps to seamlessly set up the environment, train the model with custom data, and subsequently test it on your own images and videos. Let's get started!

<div align="center">
<img align="center" src="/Results/shortvideo.gif" alt="5 Footballinfo Submodules" width = 640px height = 600px>
</div>

# Download full project [SoccerPlayer](https://github.com/tariqeee/SoccerPlayer ) here!
You have to Unzip the folder and then follow the below step 


# Pre-requisites
Make sure you have [conda](https://docs.conda.io/en/latest/
) and [pip](https://pip.pypa.io/en/stable/installation/
) installed on your machine. These are essential for creating virtual environments and installing necessary packages respectively.

You can see this video [How to Install Anaconda on Windows 10 (2022) - YouTube](https://www.youtube.com/watch?v=UTqOXwAi1pE) 

# Setup and Installation
## Step-1 
Type "cmd" on the Address bar and hit Enter.
Then follow the next step. 

Make sure that Conda is available on your PC type "conda"



## Step-2
1st Type this code first to create a conda environment
```python
conda create -n SoccerPlayer python

```

2nd active condo on your machine 

```python
conda activate SoccerPlayer
```


3rd [pip](https://pip.pypa.io/en/stable/) install YOLOv8 library  on your machine 
```bash
pip install ultralytics

```
<div align="center">
<img align="center" src="/Results/T1.jpg" alt="5 Footballinfo Submodules" width = 640px height = 600px>
</div>


4th [pip](https://pip.pypa.io/en/stable/) install PyTorch library  on your machine 
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```


## Installation

Use the package [requirements.txt](https://github.com/tariqeee/SoccerPlayer/requirements.txt) to install requirement libraries.

```bash
pip install -r requirements.txt
```

## Step-3
If you would like to train this data yourself you can do it.
Download the dataset [Dataset](https://drive.google.com/drive/folders/1iXPzVS1rNOokL2BfgfW8Vn6RWxyC-eb_?usp=sharing) 5GB storage required 

Now Start the training YOLOv8 custom model with custom data

```python
yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8n.pt imgsz=640

```
If facing any memory error use this code 

```python
yolo task=detect mode=train epochs=200 data=data_custom.yaml model=yolov8n.pt imgsz=640 batch=6

```
## After completing the training test the model.


## Now Testing my model Make sure you source files is correct. 

```python
# For 'Image' Testing use this command


yolo task=detect mode=predict model=predict model=best.pt show=True conf=0.5 source=T1.jpg imgsz=1400

# For 'Video' Testing use the command 

yolo task=detect mode=predict model=predict model=best.pt show=True conf=0.4 source=shortvideo.mp4 line_thickness=1 imgsz=1080


#For Live webcam testing use this command

yolo task=detect mode=predict model=predict model=best.pt show=True conf=0.5 source=0 line_thickness=1



# If we want to save the annotation file of the test image run this command

yolo task=detect mode=predict model=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg save_txt=True

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
